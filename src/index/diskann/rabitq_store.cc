// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "index/diskann/rabitq_store.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/index_io.h>

#include "diskann/utils.h"

namespace knowhere {
namespace {

constexpr size_t kBuildBlockBytes = 32UL * 1024 * 1024;

size_t
BlockRows(size_t dim) {
    if (dim == 0) {
        throw std::invalid_argument("RaBitQ sidecar dimension must be positive");
    }
    return std::max<size_t>(1, kBuildBlockBytes / (dim * sizeof(float)));
}

template <typename Fn>
void
ForEachFloatBinBlock(const std::string& data_path, size_t rows, size_t dim, Fn&& fn) {
    std::ifstream input(data_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open RaBitQ source data: " + data_path);
    }

    input.seekg(2 * sizeof(uint32_t), std::ios::beg);
    const size_t block_rows = BlockRows(dim);
    std::vector<float> block(block_rows * dim);
    size_t row_offset = 0;
    while (row_offset < rows) {
        const size_t current_rows = std::min(block_rows, rows - row_offset);
        const size_t current_values = current_rows * dim;
        input.read(reinterpret_cast<char*>(block.data()), current_values * sizeof(float));
        if (!input) {
            throw std::runtime_error("short read while building RaBitQ sidecar from: " + data_path);
        }
        fn(block.data(), current_rows);
        row_offset += current_rows;
    }
}

class RaBitQApproxDistanceComputer final : public diskann::ApproxDistanceComputer {
 public:
    RaBitQApproxDistanceComputer(const faiss::IndexPreTransform* pretransform, const faiss::IndexRaBitQ* rabitq,
                                 uint8_t query_bits)
        : pretransform_(pretransform),
          distance_computer_(rabitq->get_quantized_distance_computer(query_bits, false)) {
    }

    void
    set_query(const float* query) override {
        const float* transformed = pretransform_->apply_chain(1, query);
        if (transformed == query) {
            transformed_query_.reset();
            distance_computer_->set_query(query);
        } else {
            transformed_query_.reset(transformed);
            distance_computer_->set_query(transformed);
        }
    }

    void
    compute_distances(const unsigned* ids, _u64 n_ids, float* distances) override {
        for (_u64 i = 0; i < n_ids; ++i) {
            distances[i] = (*distance_computer_)(ids[i]);
        }
    }

 private:
    const faiss::IndexPreTransform* pretransform_;
    std::unique_ptr<faiss::FlatCodesDistanceComputer> distance_computer_;
    std::unique_ptr<const float[]> transformed_query_;
};

}  // namespace

std::string
RaBitQStore::SidecarFilename(const std::string& index_prefix) {
    return index_prefix + "_rabitq.index";
}

void
RaBitQStore::BuildFromFloatBin(const std::string& data_path, const std::string& sidecar_path, uint8_t rbq_bits) {
    if (rbq_bits < 1 || rbq_bits > 9) {
        throw std::invalid_argument("RaBitQ database bits must be in [1, 9]");
    }

    size_t rows = 0;
    size_t dim = 0;
    diskann::get_bin_metadata(data_path, rows, dim);
    if (rows == 0 || dim == 0 || dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid RaBitQ source metadata");
    }

    const auto expected_size = 2 * sizeof(uint32_t) + rows * dim * sizeof(float);
    if (std::filesystem::file_size(data_path) != expected_size) {
        throw std::runtime_error("RaBitQ source file size does not match float32 metadata");
    }

    auto rotation = std::make_unique<faiss::RandomRotationMatrix>(static_cast<int>(dim), static_cast<int>(dim));
    rotation->init(12345);

    std::vector<double> sums(dim, 0.0);
    ForEachFloatBinBlock(data_path, rows, dim, [&](const float* block, size_t block_rows) {
        for (size_t i = 0; i < block_rows; ++i) {
            const float* row = block + i * dim;
            for (size_t j = 0; j < dim; ++j) {
                sums[j] += row[j];
            }
        }
    });

    std::vector<float> mean(dim);
    for (size_t j = 0; j < dim; ++j) {
        mean[j] = static_cast<float>(sums[j] / static_cast<double>(rows));
    }
    std::vector<float> rotated_center(dim);
    rotation->apply_noalloc(1, mean.data(), rotated_center.data());

    auto rabitq = std::make_unique<faiss::IndexRaBitQ>(static_cast<faiss::idx_t>(dim), faiss::METRIC_L2, rbq_bits);
    rabitq->center = std::move(rotated_center);
    rabitq->qb = 0;
    rabitq->centered = false;
    rabitq->is_trained = true;
    auto pretransform = std::make_unique<faiss::IndexPreTransform>(rotation.get(), rabitq.get());
    pretransform->own_fields = true;
    rotation.release();
    rabitq.release();

    ForEachFloatBinBlock(data_path, rows, dim, [&](const float* block, size_t block_rows) {
        pretransform->add(static_cast<faiss::idx_t>(block_rows), block);
    });
    if (pretransform->ntotal != static_cast<faiss::idx_t>(rows)) {
        throw std::runtime_error("RaBitQ sidecar point count mismatch after encoding");
    }

    const std::string temporary_path = sidecar_path + ".tmp";
    std::error_code error;
    std::filesystem::remove(temporary_path, error);
    try {
        faiss::write_index(pretransform.get(), temporary_path.c_str());
        std::filesystem::rename(temporary_path, sidecar_path);
    } catch (...) {
        std::filesystem::remove(temporary_path, error);
        throw;
    }
}

RaBitQStore::RaBitQStore(const std::string& sidecar_path) : index_(faiss::read_index(sidecar_path.c_str())) {
    Validate();
}

RaBitQStore::~RaBitQStore() = default;

void
RaBitQStore::Validate() {
    pretransform_ = dynamic_cast<const faiss::IndexPreTransform*>(index_.get());
    if (pretransform_ == nullptr || pretransform_->chain.size() != 1) {
        throw std::runtime_error("DiskANN RaBitQ sidecar must be an IndexPreTransform with one transform");
    }
    rotation_ = dynamic_cast<const faiss::RandomRotationMatrix*>(pretransform_->chain[0]);
    if (rotation_ == nullptr || !rotation_->is_trained || rotation_->d_in != rotation_->d_out) {
        throw std::runtime_error("DiskANN RaBitQ sidecar has an invalid random rotation");
    }
    rabitq_ = dynamic_cast<const faiss::IndexRaBitQ*>(pretransform_->index);
    if (rabitq_ == nullptr || !pretransform_->is_trained || !rabitq_->is_trained) {
        throw std::runtime_error("DiskANN RaBitQ sidecar has an invalid RaBitQ leaf");
    }
    if (pretransform_->metric_type != faiss::METRIC_L2 || rabitq_->metric_type != faiss::METRIC_L2 ||
        pretransform_->d != rotation_->d_in || rabitq_->d != rotation_->d_out ||
        pretransform_->ntotal != rabitq_->ntotal) {
        throw std::runtime_error("DiskANN RaBitQ sidecar metadata is inconsistent");
    }
    if (rabitq_->rabitq.nb_bits < 1 || rabitq_->rabitq.nb_bits > 9 ||
        rabitq_->center.size() != static_cast<size_t>(rabitq_->d) ||
        rabitq_->codes.size() != static_cast<size_t>(rabitq_->ntotal) * rabitq_->code_size) {
        throw std::runtime_error("DiskANN RaBitQ sidecar code storage is inconsistent");
    }
}

std::unique_ptr<diskann::ApproxDistanceComputer>
RaBitQStore::CreateDistanceComputer(uint8_t query_bits) const {
    if (query_bits > 8) {
        throw std::invalid_argument("RaBitQ query bits must be in [0, 8]");
    }
    if (Bits() > 1 && query_bits > 0) {
        throw std::invalid_argument("quantized RaBitQ queries currently require 1-bit database codes");
    }
    return std::make_unique<RaBitQApproxDistanceComputer>(pretransform_, rabitq_, query_bits);
}

int64_t
RaBitQStore::Count() const {
    return pretransform_->ntotal;
}

int64_t
RaBitQStore::Dimension() const {
    return pretransform_->d;
}

uint8_t
RaBitQStore::Bits() const {
    return static_cast<uint8_t>(rabitq_->rabitq.nb_bits);
}

size_t
RaBitQStore::CodeSize() const {
    return rabitq_->code_size;
}

size_t
RaBitQStore::MemorySize() const {
    return rabitq_->codes.size() * sizeof(uint8_t) + rabitq_->center.size() * sizeof(float) +
           rotation_->A.size() * sizeof(float) + rotation_->b.size() * sizeof(float);
}

}  // namespace knowhere
