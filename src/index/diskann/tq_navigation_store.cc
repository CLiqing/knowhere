// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "index/diskann/tq_navigation_store.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/index_io.h>

#include "diskann/utils.h"

namespace knowhere {
namespace {

constexpr size_t kBuildBlockBytes = 32UL * 1024 * 1024;

faiss::ScalarQuantizer::QuantizerType
QuantizerTypeForBits(uint8_t bits) {
    switch (bits) {
        case 2:
            return faiss::ScalarQuantizer::QT_2bit_tqmse;
        case 3:
            return faiss::ScalarQuantizer::QT_3bit_tqmse;
        case 4:
            return faiss::ScalarQuantizer::QT_4bit_tqmse;
        default:
            throw std::invalid_argument("TQ-MSE navigation bits must be in [2, 4]");
    }
}

uint8_t
BitsForQuantizerType(faiss::ScalarQuantizer::QuantizerType qtype) {
    switch (qtype) {
        case faiss::ScalarQuantizer::QT_2bit_tqmse:
            return 2;
        case faiss::ScalarQuantizer::QT_3bit_tqmse:
            return 3;
        case faiss::ScalarQuantizer::QT_4bit_tqmse:
            return 4;
        default:
            throw std::runtime_error("DiskANN TQ sidecar has an unsupported scalar quantizer type");
    }
}

size_t
BlockRows(size_t dim) {
    if (dim == 0) {
        throw std::invalid_argument("TQ navigation sidecar dimension must be positive");
    }
    return std::max<size_t>(1, kBuildBlockBytes / (dim * sizeof(float)));
}

template <typename Fn>
void
ForEachFloatBinBlock(const std::string& data_path, size_t rows, size_t dim, Fn&& fn) {
    std::ifstream input(data_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open TQ navigation source data: " + data_path);
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
            throw std::runtime_error("short read while building TQ navigation sidecar from: " + data_path);
        }
        fn(block.data(), current_rows);
        row_offset += current_rows;
    }
}

void
ApplyRotationSingleQuery(const faiss::RandomRotationMatrix* rotation, const float* x, float* xt) {
    const int d = rotation->d_in;
    const float* matrix = rotation->A.data();
    for (int j = 0; j < d; ++j) {
        const float* column = matrix + static_cast<size_t>(j) * d;
        float accumulator = rotation->have_bias ? rotation->b[j] : 0.0f;
#pragma omp simd reduction(+ : accumulator)
        for (int k = 0; k < d; ++k) {
            accumulator += column[k] * x[k];
        }
        xt[j] = accumulator;
    }
}

class TQApproxDistanceComputer final : public diskann::ApproxDistanceComputer {
 public:
    TQApproxDistanceComputer(const faiss::RandomRotationMatrix* rotation,
                             const faiss::IndexScalarQuantizer* scalar_quantizer)
        : rotation_(rotation),
          distance_computer_(scalar_quantizer->get_FlatCodesDistanceComputer()),
          transformed_query_(static_cast<size_t>(rotation->d_out)) {
    }

    void
    set_query(const float* query) override {
        ApplyRotationSingleQuery(rotation_, query, transformed_query_.data());
        distance_computer_->set_query(transformed_query_.data());
    }

    void
    compute_distances(const unsigned* ids, _u64 n_ids, float* distances, float, bool,
                      diskann::QueryStats*) override {
        _u64 i = 0;
        for (; i + 4 <= n_ids; i += 4) {
            distance_computer_->distances_batch_4(ids[i], ids[i + 1], ids[i + 2], ids[i + 3], distances[i],
                                                  distances[i + 1], distances[i + 2], distances[i + 3]);
        }
        for (; i < n_ids; ++i) {
            distances[i] = (*distance_computer_)(ids[i]);
        }
    }

 private:
    const faiss::RandomRotationMatrix* rotation_;
    std::unique_ptr<faiss::FlatCodesDistanceComputer> distance_computer_;
    std::vector<float> transformed_query_;
};

}  // namespace

std::string
TQNavigationStore::SidecarFilename(const std::string& index_prefix) {
    return index_prefix + "_tqmse.index";
}

void
TQNavigationStore::BuildFromFloatBin(const std::string& data_path, const std::string& sidecar_path, uint8_t bits) {
    size_t rows = 0;
    size_t dim = 0;
    diskann::get_bin_metadata(data_path, rows, dim);
    if (rows == 0 || dim == 0 || dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid TQ navigation source metadata");
    }
    const auto expected_size = 2 * sizeof(uint32_t) + rows * dim * sizeof(float);
    if (std::filesystem::file_size(data_path) != expected_size) {
        throw std::runtime_error("TQ navigation source file size does not match float32 metadata");
    }

    auto rotation = std::make_unique<faiss::RandomRotationMatrix>(static_cast<int>(dim), static_cast<int>(dim));
    rotation->init(12345);
    auto scalar_quantizer = std::make_unique<faiss::IndexScalarQuantizer>(
        static_cast<faiss::idx_t>(dim), QuantizerTypeForBits(bits), faiss::METRIC_L2);
    auto pretransform = std::make_unique<faiss::IndexPreTransform>(rotation.get(), scalar_quantizer.get());
    pretransform->own_fields = true;
    rotation.release();
    scalar_quantizer.release();

    bool trained = false;
    ForEachFloatBinBlock(data_path, rows, dim, [&](const float* block, size_t block_rows) {
        if (!trained) {
            pretransform->train(static_cast<faiss::idx_t>(block_rows), block);
            trained = true;
        }
        pretransform->add(static_cast<faiss::idx_t>(block_rows), block);
    });
    if (!pretransform->is_trained || pretransform->ntotal != static_cast<faiss::idx_t>(rows)) {
        throw std::runtime_error("TQ navigation sidecar count mismatch after encoding");
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

TQNavigationStore::TQNavigationStore(const std::string& sidecar_path) : index_(faiss::read_index(sidecar_path.c_str())) {
    Validate();
}

TQNavigationStore::~TQNavigationStore() = default;

void
TQNavigationStore::Validate() {
    pretransform_ = dynamic_cast<const faiss::IndexPreTransform*>(index_.get());
    if (pretransform_ == nullptr || pretransform_->chain.size() != 1 || !pretransform_->is_trained) {
        throw std::runtime_error("DiskANN TQ sidecar must be a trained IndexPreTransform with one transform");
    }
    rotation_ = dynamic_cast<const faiss::RandomRotationMatrix*>(pretransform_->chain[0]);
    if (rotation_ == nullptr || !rotation_->is_trained || rotation_->d_in <= 0 ||
        rotation_->d_in != rotation_->d_out) {
        throw std::runtime_error("DiskANN TQ sidecar has an invalid random rotation");
    }
    const auto dimension = static_cast<size_t>(rotation_->d_in);
    if (rotation_->A.size() != dimension * dimension ||
        (rotation_->have_bias ? rotation_->b.size() != dimension : !rotation_->b.empty())) {
        throw std::runtime_error("DiskANN TQ sidecar rotation storage is inconsistent");
    }
    scalar_quantizer_ = dynamic_cast<const faiss::IndexScalarQuantizer*>(pretransform_->index);
    if (scalar_quantizer_ == nullptr || !scalar_quantizer_->is_trained ||
        pretransform_->metric_type != faiss::METRIC_L2 || scalar_quantizer_->metric_type != faiss::METRIC_L2 ||
        pretransform_->d != rotation_->d_in || scalar_quantizer_->d != rotation_->d_out ||
        pretransform_->ntotal != scalar_quantizer_->ntotal) {
        throw std::runtime_error("DiskANN TQ sidecar metadata is inconsistent");
    }
    BitsForQuantizerType(scalar_quantizer_->sq.qtype);
    if (scalar_quantizer_->ntotal < 0 || scalar_quantizer_->code_size == 0 ||
        static_cast<size_t>(scalar_quantizer_->ntotal) >
            std::numeric_limits<size_t>::max() / scalar_quantizer_->code_size ||
        scalar_quantizer_->codes.size() !=
            static_cast<size_t>(scalar_quantizer_->ntotal) * scalar_quantizer_->code_size) {
        throw std::runtime_error("DiskANN TQ sidecar code storage is inconsistent");
    }
}

std::unique_ptr<diskann::ApproxDistanceComputer>
TQNavigationStore::CreateDistanceComputer() const {
    return std::make_unique<TQApproxDistanceComputer>(rotation_, scalar_quantizer_);
}

int64_t
TQNavigationStore::Count() const {
    return pretransform_->ntotal;
}

int64_t
TQNavigationStore::Dimension() const {
    return pretransform_->d;
}

uint8_t
TQNavigationStore::Bits() const {
    return BitsForQuantizerType(scalar_quantizer_->sq.qtype);
}

size_t
TQNavigationStore::CodeSize() const {
    return scalar_quantizer_->code_size;
}

size_t
TQNavigationStore::MemorySize() const {
    return scalar_quantizer_->codes.size() + scalar_quantizer_->sq.trained.size() * sizeof(float) +
           rotation_->A.size() * sizeof(float) + rotation_->b.size() * sizeof(float);
}

}  // namespace knowhere
