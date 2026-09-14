// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "index/diskann/tq_navigation_store.h"

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include <filesystem>
#include <fstream>

#include "index/turboquant/turboquant_utils.h"

namespace knowhere {
namespace {
uint32_t
MetricId(const std::string& metric) {
    if (metric == "L2")
        return 0;
    if (metric == "IP")
        return 1;
    if (metric == "COSINE")
        return 2;
    throw std::invalid_argument("unsupported TQ navigation metric");
}
std::vector<uint32_t>
Header(bool mse, int bits, const std::string& metric) {
    return {0x54514e31, 1, uint32_t(mse), uint32_t(bits), MetricId(metric)};
}
class TQNavigationDistanceComputer : public diskann::NavigationDistanceComputer {
 public:
    TQNavigationDistanceComputer(const faiss::IndexPreTransform* chain, int qb, bool int_qjl)
        : rr_(static_cast<const faiss::RandomRotationMatrix*>(chain->chain[0])),
          dc_(chain->index->get_distance_computer()),
          rotated_(chain->d) {
        if (auto* full = dynamic_cast<faiss::ScalarQuantizer::TurboQuantRefine::DistanceComputer*>(dc_.get())) {
            full->configure(qb, int_qjl);
        } else if (qb != 0 || int_qjl) {
            throw std::invalid_argument("TQ-MSE navigation has no query quantization parameters");
        }
    }
    void
    set_query(const float* q) override {
        // GEMV without a per-query BLAS thread team; same matrix/orientation as Faiss RR.
        for (int i = 0; i < rr_->d_out; ++i) {
            float sum = 0;
            const auto* row = rr_->A.data() + size_t(i) * rr_->d_in;
#pragma omp simd reduction(+ : sum)
            for (int j = 0; j < rr_->d_in; ++j) sum += row[j] * q[j];
            rotated_[i] = sum;
        }
        dc_->set_query(rotated_.data());
    }
    void
    compute_distances(const unsigned* ids, _u64 n, float* distances) override {
        _u64 i = 0;
        for (; i + 4 <= n; i += 4) {
            dc_->distances_batch_4(ids[i], ids[i + 1], ids[i + 2], ids[i + 3], distances[i], distances[i + 1],
                                   distances[i + 2], distances[i + 3]);
        }
        for (; i < n; ++i) distances[i] = (*dc_)(ids[i]);
    }

 private:
    const faiss::RandomRotationMatrix* rr_;
    std::unique_ptr<faiss::DistanceComputer> dc_;
    std::vector<float> rotated_;
};
}  // namespace

void
TQNavigationStore::Build(const std::string& source, const std::string& file, bool mse, int bits,
                         const std::string& metric, double budget_gb) {
    if (mse && metric == "L2")
        throw std::invalid_argument("TQ-MSE ordinary L2 is unsupported");
    std::ifstream input(source, std::ios::binary);
    uint32_t rows = 0, dim = 0;
    input.read(reinterpret_cast<char*>(&rows), 4);
    input.read(reinterpret_cast<char*>(&dim), 4);
    if (!input || rows == 0 || dim == 0 || dim > 65536 ||
        std::filesystem::file_size(source) != 8 + uint64_t(rows) * dim * sizeof(float)) {
        throw std::invalid_argument("invalid TQ navigation float32 source");
    }
    auto sq =
        std::make_unique<faiss::IndexScalarQuantizer>(dim, turboquant::QuantizerType(mse, bits), faiss::METRIC_L2);
    const uint64_t estimated =
        uint64_t(rows) * sq->code_size + uint64_t(dim) * dim * sizeof(float) + sq->sq.trained.size() * 4;
    if (budget_gb > 0 && estimated > budget_gb * 1024 * 1024 * 1024) {
        throw std::invalid_argument("TQ navigation memory budget is insufficient");
    }
    auto rr = std::make_unique<faiss::RandomRotationMatrix>(dim, dim);
    rr->init(12345);
    auto chain = std::make_unique<faiss::IndexPreTransform>(rr.get(), sq.get());
    chain->own_fields = true;
    rr.release();
    sq.release();
    const size_t batch = std::max<size_t>(1, (32 * 1024 * 1024) / (dim * sizeof(float)));
    std::vector<float> values(std::min<size_t>(batch, rows) * dim);
    for (size_t offset = 0; offset < rows; offset += batch) {
        const auto n = std::min<size_t>(batch, rows - offset);
        input.read(reinterpret_cast<char*>(values.data()), n * dim * sizeof(float));
        if (!input)
            throw std::runtime_error("short TQ navigation source read");
        if (offset == 0)
            chain->train(n, values.data());
        chain->add(n, values.data());
    }
    const auto tmp = file + ".tmp";
    if (std::filesystem::exists(tmp) || std::filesystem::exists(file)) {
        throw std::runtime_error("TQ sidecar destination already exists");
    }
    try {
        {
            faiss::FileIOWriter writer(tmp.c_str());
            const auto header = Header(mse, bits, metric);
            writer(header.data(), sizeof(uint32_t), header.size());
            faiss::write_index(chain.get(), &writer);
        }
        std::filesystem::rename(tmp, file);
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(tmp, error);
        throw;
    }
}

TQNavigationStore::TQNavigationStore(const std::string& file, bool mse, int bits, const std::string& metric)
    : mse_(mse) {
    faiss::FileIOReader reader(file.c_str());
    auto expected = Header(mse, bits, metric), actual = expected;
    if (reader(actual.data(), sizeof(uint32_t), actual.size()) != actual.size() || actual != expected) {
        throw std::invalid_argument("TQ navigation sidecar version/variant/metric/bits mismatch");
    }
    index_.reset(faiss::read_index(&reader));
    auto* chain = dynamic_cast<const faiss::IndexPreTransform*>(index_.get());
    if (!chain || !chain->is_trained || chain->chain.size() != 1 || chain->d <= 0 || chain->ntotal <= 0) {
        throw std::invalid_argument("invalid TQ navigation transform chain");
    }
    auto* sq = dynamic_cast<const faiss::IndexScalarQuantizer*>(chain->index);
    auto* rr = dynamic_cast<const faiss::RandomRotationMatrix*>(chain->chain[0]);
    faiss::ScalarQuantizer expected_sq(chain->d, turboquant::QuantizerType(mse, bits));
    if (!sq || !rr || !sq->is_trained || !rr->is_trained || rr->have_bias || rr->d_in != chain->d ||
        rr->d_out != chain->d || rr->A.size() != size_t(chain->d) * chain->d || sq->d != chain->d ||
        sq->ntotal != chain->ntotal || sq->sq.qtype != expected_sq.qtype || sq->code_size != expected_sq.code_size ||
        sq->sq.code_size != expected_sq.code_size || sq->sq.d != chain->d || sq->metric_type != faiss::METRIC_L2 ||
        chain->metric_type != faiss::METRIC_L2 || sq->codes.size() != size_t(sq->ntotal) * sq->code_size) {
        throw std::invalid_argument("inconsistent TQ navigation sidecar");
    }
}
TQNavigationStore::~TQNavigationStore() = default;
std::unique_ptr<diskann::NavigationDistanceComputer>
TQNavigationStore::CreateDistanceComputer(int qb, bool int_qjl) const {
    return std::make_unique<TQNavigationDistanceComputer>(static_cast<const faiss::IndexPreTransform*>(index_.get()),
                                                          qb, int_qjl);
}
int64_t
TQNavigationStore::Count() const {
    return index_->ntotal;
}
int64_t
TQNavigationStore::Dimension() const {
    return index_->d;
}
size_t
TQNavigationStore::MemorySize() const {
    auto* chain = static_cast<const faiss::IndexPreTransform*>(index_.get());
    auto* sq = static_cast<const faiss::IndexScalarQuantizer*>(chain->index);
    return sq->codes.size() + sq->sq.trained.size() * sizeof(float) + size_t(chain->d) * chain->d * sizeof(float);
}
}  // namespace knowhere
