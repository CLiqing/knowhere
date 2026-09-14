// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {
void
NormalizeTQRows(float* data, int rows, int dim) {
    for (int i = 0; i < rows; ++i) {
        float norm = 0;
        for (int j = 0; j < dim; ++j) {
            norm += data[i * dim + j] * data[i * dim + j];
        }
        if (norm > 0) {
            for (int j = 0; j < dim; ++j) {
                data[i * dim + j] /= std::sqrt(norm);
            }
        }
    }
}
}  // namespace

TEST_CASE("HNSW TQ metrics and query parameters match the Faiss codec", "[turboquant][tq_acceptance]") {
    constexpr int nb = 128, nq = 5, dim = 33;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (bool mse : {GENERATE(false, true)}) {
        for (const std::string metric : {"L2", "IP", "COSINE"}) {
            if (mse && metric == "L2") {
                continue;
            }
            for (int bits : (mse ? std::vector<int>{1, 2, 3, 4, 8} : std::vector<int>{2, 3, 4, 5})) {
                CAPTURE(mse, metric, bits);
                auto base = GenDataSet(nb, dim, 471);
                auto query = GenDataSet(nq, dim, 892);
                auto* xb = const_cast<float*>(static_cast<const float*>(base->GetTensor()));
                auto* xq = const_cast<float*>(static_cast<const float*>(query->GetTensor()));
                NormalizeTQRows(xb, nb, dim);
                NormalizeTQRows(xq, nq, dim);
                if (!mse || metric == "COSINE") {
                    for (int i = 0; i < nb; ++i) {
                        for (int j = 0; j < dim; ++j) xb[i * dim + j] *= 0.25f + (i % 9);
                    }
                    for (int i = 0; i < nq; ++i) {
                        for (int j = 0; j < dim; ++j) xq[i * dim + j] *= 0.5f + i;
                    }
                }
                std::fill(xb, xb + dim, 0);
                std::fill(xq, xq + dim, 0);
                const std::vector<float> original_base(xb, xb + nb * dim);
                const std::vector<float> original_query(xq, xq + nq * dim);
                std::vector<float> ref_base(xb, xb + nb * dim), ref_query(xq, xq + nq * dim);
                if (metric == "COSINE") {
                    NormalizeTQRows(ref_base.data(), nb, dim);
                    NormalizeTQRows(ref_query.data(), nq, dim);
                }
                faiss::RandomRotationMatrix rr(dim, dim);
                rr.init(12345);
                std::vector<float> rb(nb * dim), rq(nq * dim);
                rr.apply_noalloc(nb, ref_base.data(), rb.data());
                rr.apply_noalloc(nq, ref_query.data(), rq.data());
                using SQ = faiss::ScalarQuantizer;
                const auto qtype =
                    mse ? (bits == 8 ? SQ::QT_8bit_tqmse : static_cast<SQ::QuantizerType>(SQ::QT_1bit_tqmse + bits - 1))
                        : static_cast<SQ::QuantizerType>(SQ::QT_2bit_tq + bits - 2);
                faiss::IndexScalarQuantizer reference(dim, qtype,
                                                      metric == "L2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT);
                reference.train(nb, rb.data());
                reference.add(nb, rb.data());
                knowhere::Json cfg = {
                    {"dim", dim}, {"metric_type", metric}, {"M", 8}, {"efConstruction", 64}, {"ef", nb},
                    {"k", nb},    {"tq_bits", bits}};
                auto index = knowhere::IndexFactory::Instance()
                                 .Create<knowhere::fp32>(mse ? knowhere::IndexEnum::INDEX_HNSW_TQMSE
                                                             : knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT,
                                                         version)
                                 .value();
                REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
                REQUIRE(std::equal(original_base.begin(), original_base.end(), xb));
                for (int qb : (mse ? std::vector<int>{0} : std::vector<int>{0, 4, 8})) {
                    CAPTURE(qb);
                    cfg["tq_query_bits"] = qb;
                    cfg["tq_int_qjl"] = qb != 0;
                    auto dc = std::unique_ptr<faiss::DistanceComputer>(reference.get_distance_computer());
                    if (!mse) {
                        dynamic_cast<SQ::TurboQuantRefine::DistanceComputer*>(dc.get())->configure(qb, qb != 0);
                    }
                    for (int k : {8, nb}) {
                        CAPTURE(k);
                        cfg["k"] = k;
                        auto result = index.Search(query, cfg, knowhere::BitsetView{});
                        REQUIRE(result.has_value());
                        REQUIRE(std::equal(original_query.begin(), original_query.end(), xq));
                        for (int q = 0; q < nq; ++q) {
                            dc->set_query(rq.data() + q * dim);
                            for (int j = 0; j < k; ++j) {
                                const auto id = result.value()->GetIds()[q * k + j];
                                REQUIRE(id >= 0);
                                const float expected = (*dc)(id);
                                const float actual = result.value()->GetDistance()[q * k + j];
                                CAPTURE(q, id, expected, actual);
                                REQUIRE(std::abs(actual - expected) <= 2e-4f * (1.0f + std::abs(expected)));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("HNSW Full TQ filtering and concurrent requests preserve query parameters", "[turboquant][tq_acceptance]") {
    constexpr int nb = 128, nq = 4, dim = 32;
    auto base = GenDataSet(nb, dim, 427);
    auto query = GenDataSet(nq, dim, 837);
    knowhere::Json cfg = {{"dim", dim}, {"metric_type", "IP"}, {"M", 8}, {"efConstruction", 64}, {"ef", nb},
                          {"k", nb},    {"tq_bits", 2}};
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT,
                                             knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
    std::vector<knowhere::DataSetPtr> reference;
    for (int qb : {0, 4, 8}) {
        cfg["tq_query_bits"] = qb;
        cfg["tq_int_qjl"] = qb != 0;
        auto result = index.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(result.has_value());
        reference.push_back(result.value());
    }
    std::vector<std::future<knowhere::expected<knowhere::DataSetPtr>>> pending;
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (int qb : {0, 4, 8}) {
            auto local_cfg = cfg;
            local_cfg["tq_query_bits"] = qb;
            local_cfg["tq_int_qjl"] = qb != 0;
            pending.push_back(std::async(std::launch::async, [&, local_cfg]() {
                return index.Search(query, local_cfg, knowhere::BitsetView{});
            }));
        }
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        auto result = pending[i].get();
        REQUIRE(result.has_value());
        for (int j = 0; j < nq * nb; ++j) {
            REQUIRE(result.value()->GetIds()[j] == reference[i % 3]->GetIds()[j]);
            REQUIRE(result.value()->GetDistance()[j] == reference[i % 3]->GetDistance()[j]);
        }
    }
    for (int excluded : {16, 112}) {
        std::vector<uint8_t> mask(nb / 8, 0);
        for (int i = 0; i < excluded; ++i) mask[i / 8] |= uint8_t(1 << (i % 8));
        knowhere::BitsetView bitset(mask.data(), nb);
        cfg["k"] = 8;
        for (int variant = 0; variant < 3; ++variant) {
            const int qb = variant * 4;
            cfg["tq_query_bits"] = qb;
            cfg["tq_int_qjl"] = qb != 0;
            auto result = index.Search(query, cfg, bitset);
            REQUIRE(result.has_value());
            for (int q = 0; q < nq; ++q) {
                for (int j = 0; j < 8; ++j) {
                    auto id = result.value()->GetIds()[q * 8 + j];
                    REQUIRE(id >= excluded);
                    REQUIRE(id < nb);
                    const auto* ids = reference[variant]->GetIds() + q * nb;
                    const auto pos = std::find(ids, ids + nb, id) - ids;
                    REQUIRE(pos < nb);
                    REQUIRE(result.value()->GetDistance()[q * 8 + j] ==
                            reference[variant]->GetDistance()[q * nb + pos]);
                }
            }
        }
    }
}

TEST_CASE("HNSW TurboQuant declares unsupported entry points", "[hnsw][turboquant][tq_acceptance]") {
    auto base = GenDataSet(128, 33, 123);
    auto query = GenDataSet(1, 33, 456);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (const auto* type : {knowhere::IndexEnum::INDEX_HNSW_TQMSE, knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT}) {
        CAPTURE(type);
        knowhere::Json cfg = {{"dim", 33},    {"metric_type", "COSINE"},
                              {"M", 8},       {"efConstruction", 32},
                              {"ef", 32},     {"k", 8},
                              {"tq_bits", 4}, {"radius", 0.5}};
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(type, version).value();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
        REQUIRE_FALSE(index.HasRawData("COSINE"));
        auto iterator = index.AnnIterator(query, cfg, knowhere::BitsetView{});
        REQUIRE_FALSE(iterator.has_value());
        REQUIRE(iterator.error() == knowhere::Status::not_implemented);
        for (int ef : {32, 128}) {
            cfg["ef"] = ef;
            auto range = index.RangeSearch(query, cfg, knowhere::BitsetView{});
            REQUIRE_FALSE(range.has_value());
            REQUIRE(range.error() == knowhere::Status::not_implemented);
        }
        if (std::string(type) == knowhere::IndexEnum::INDEX_HNSW_TQMSE) {
            cfg["tq_query_bits"] = 4;
            REQUIRE_FALSE(index.Search(query, cfg, knowhere::BitsetView{}).has_value());
            cfg["tq_query_bits"] = 0;
            cfg["tq_int_qjl"] = true;
            REQUIRE_FALSE(index.Search(query, cfg, knowhere::BitsetView{}).has_value());
            cfg["tq_int_qjl"] = false;
            cfg["metric_type"] = "L2";
            auto l2 = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(type, version).value();
            REQUIRE(l2.Build(base, cfg) == knowhere::Status::invalid_metric_type);
        }
    }
}

TEST_CASE("HNSW Full TurboQuant applies RR and survives serialization", "[hnsw][turboquant]") {
    constexpr int64_t kNb = 512;
    constexpr int64_t kNq = 16;
    constexpr int64_t kDim = 32;
    constexpr int64_t kTopK = 10;

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 84);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    for (const std::string metric : {knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE}) {
        knowhere::Json cfg;
        cfg[knowhere::meta::DIM] = kDim;
        cfg[knowhere::meta::METRIC_TYPE] = metric;
        cfg[knowhere::indexparam::HNSW_M] = 8;
        cfg[knowhere::indexparam::EFCONSTRUCTION] = 64;
        cfg[knowhere::indexparam::EF] = 64;
        cfg[knowhere::meta::TOPK] = kTopK;
        cfg[knowhere::indexparam::TURBOQUANT_BITS] = 2;
        cfg[knowhere::indexparam::TURBOQUANT_QUERY_BITS] = 0;
        cfg[knowhere::indexparam::TURBOQUANT_INT_QJL] = false;

        auto index_result = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT, version);
        REQUIRE(index_result.has_value());
        auto index = index_result.value();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);

        auto float_result = index.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(float_result.has_value());
        REQUIRE(float_result.value()->GetRows() == kNq);
        REQUIRE(float_result.value()->GetDim() == kTopK);
        const auto* float_ids = float_result.value()->GetIds();
        const auto* float_distances = float_result.value()->GetDistance();
        for (int64_t i = 0; i < kNq * kTopK; ++i) {
            REQUIRE(float_ids[i] >= 0);
            REQUIRE(float_ids[i] < kNb);
            REQUIRE(std::isfinite(float_distances[i]));
        }

        cfg[knowhere::indexparam::TURBOQUANT_QUERY_BITS] = 4;
        cfg[knowhere::indexparam::TURBOQUANT_INT_QJL] = true;
        auto integer_result = index.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(integer_result.has_value());

        knowhere::BinarySet binary_set;
        REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
        auto loaded_result = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT, version);
        REQUIRE(loaded_result.has_value());
        auto loaded = loaded_result.value();
        REQUIRE(loaded.Deserialize(binary_set, cfg) == knowhere::Status::success);
        auto loaded_search = loaded.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(loaded_search.has_value());

        const auto* integer_ids = integer_result.value()->GetIds();
        const auto* loaded_ids = loaded_search.value()->GetIds();
        for (int64_t i = 0; i < kNq * kTopK; ++i) {
            REQUIRE(integer_ids[i] == loaded_ids[i]);
        }
    }
}

TEST_CASE("HNSW TurboQuant rejects inconsistent integer QJL settings", "[hnsw][turboquant]") {
    constexpr int64_t kDim = 32;
    auto base = GenDataSet(128, kDim, 43);
    auto query = GenDataSet(1, kDim, 85);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    knowhere::Json cfg;
    cfg[knowhere::meta::DIM] = kDim;
    cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    cfg[knowhere::indexparam::HNSW_M] = 8;
    cfg[knowhere::indexparam::EFCONSTRUCTION] = 32;
    cfg[knowhere::indexparam::EF] = 32;
    cfg[knowhere::meta::TOPK] = 10;
    cfg[knowhere::indexparam::TURBOQUANT_BITS] = 2;

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT, version)
                     .value();
    REQUIRE(index.Build(base, cfg) == knowhere::Status::success);

    cfg[knowhere::indexparam::TURBOQUANT_QUERY_BITS] = 0;
    cfg[knowhere::indexparam::TURBOQUANT_INT_QJL] = true;
    auto result = index.Search(query, cfg, knowhere::BitsetView{});
    REQUIRE_FALSE(result.has_value());

    auto extra = GenDataSet(1, kDim, 86);
    REQUIRE(index.Add(extra, cfg) == knowhere::Status::not_implemented);
}

TEST_CASE("HNSW Full TurboQuant rejects refine with outer RR", "[hnsw][turboquant]") {
    constexpr int64_t kDim = 32;
    auto base = GenDataSet(128, kDim, 46);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    knowhere::Json cfg;
    cfg[knowhere::meta::DIM] = kDim;
    cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    cfg[knowhere::indexparam::HNSW_M] = 8;
    cfg[knowhere::indexparam::EFCONSTRUCTION] = 32;
    cfg[knowhere::indexparam::TURBOQUANT_BITS] = 4;
    cfg[knowhere::indexparam::REFINE] = true;
    cfg[knowhere::indexparam::REFINE_TYPE] = "FP32";

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT, version)
                     .value();
    REQUIRE(index.Build(base, cfg) == knowhere::Status::not_implemented);
}

TEST_CASE("HNSW TurboQuant supports all full-code bit widths", "[hnsw][turboquant]") {
    constexpr int64_t kDim = 32;
    auto base = GenDataSet(256, kDim, 44);
    auto query = GenDataSet(4, kDim, 86);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    for (int bits : {2, 3, 4, 5}) {
        knowhere::Json cfg;
        cfg[knowhere::meta::DIM] = kDim;
        cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        cfg[knowhere::indexparam::HNSW_M] = 8;
        cfg[knowhere::indexparam::EFCONSTRUCTION] = 32;
        cfg[knowhere::indexparam::EF] = 32;
        cfg[knowhere::meta::TOPK] = 10;
        cfg[knowhere::indexparam::TURBOQUANT_BITS] = bits;

        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT, version)
                         .value();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
        auto result = index.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(result.has_value());
    }
}

TEST_CASE("HNSW TurboQuant MSE applies RR and survives serialization", "[hnsw][turboquant][tqmse]") {
    constexpr int64_t kNb = 512;
    constexpr int64_t kNq = 16;
    constexpr int64_t kDim = 32;
    constexpr int64_t kTopK = 10;
    auto base = GenDataSet(kNb, kDim, 45);
    auto query = GenDataSet(kNq, kDim, 87);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    for (int bits : {1, 2, 3, 4, 8}) {
        CAPTURE(bits);
        knowhere::Json cfg;
        cfg[knowhere::meta::DIM] = kDim;
        cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        cfg[knowhere::indexparam::HNSW_M] = 8;
        cfg[knowhere::indexparam::EFCONSTRUCTION] = 64;
        cfg[knowhere::indexparam::EF] = 64;
        cfg[knowhere::meta::TOPK] = kTopK;
        cfg[knowhere::indexparam::TURBOQUANT_BITS] = bits;

        auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_HNSW_TQMSE, version);
        REQUIRE(created.has_value());
        auto index = created.value();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
        auto result = index.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(result.has_value());

        knowhere::BinarySet binary_set;
        REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
        auto loaded = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_TQMSE, version)
                          .value();
        REQUIRE(loaded.Deserialize(binary_set, cfg) == knowhere::Status::success);
        auto loaded_result = loaded.Search(query, cfg, knowhere::BitsetView{});
        REQUIRE(loaded_result.has_value());

        const auto* ids = result.value()->GetIds();
        const auto* loaded_ids = loaded_result.value()->GetIds();
        for (int64_t i = 0; i < kNq * kTopK; ++i) {
            REQUIRE(ids[i] >= 0);
            REQUIRE(ids[i] < kNb);
            REQUIRE(std::isfinite(result.value()->GetDistance()[i]));
            REQUIRE(ids[i] == loaded_ids[i]);
            REQUIRE(result.value()->GetDistance()[i] == loaded_result.value()->GetDistance()[i]);
        }
    }
}

TEST_CASE("HNSW TurboQuant validates variant-specific bit widths", "[hnsw][turboquant][tqmse]") {
    constexpr int64_t kDim = 33;
    auto base = GenDataSet(128, kDim, 46);
    auto query = GenDataSet(4, kDim, 88);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    knowhere::Json cfg = {{knowhere::meta::DIM, kDim},       {knowhere::meta::METRIC_TYPE, knowhere::metric::COSINE},
                          {knowhere::indexparam::HNSW_M, 8}, {knowhere::indexparam::EFCONSTRUCTION, 32},
                          {knowhere::indexparam::EF, 32},    {knowhere::meta::TOPK, 10}};
    for (const auto* type : {knowhere::IndexEnum::INDEX_HNSW_TQMSE, knowhere::IndexEnum::INDEX_HNSW_TURBOQUANT}) {
        for (int bits : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
            CAPTURE(type, bits);
            cfg[knowhere::indexparam::TURBOQUANT_BITS] = bits;
            auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(type, version);
            REQUIRE(created.has_value());
            auto index = created.value();
            const bool mse = std::string(type) == knowhere::IndexEnum::INDEX_HNSW_TQMSE;
            const bool valid =
                mse ? (bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 8) : (bits >= 2 && bits <= 5);
            const auto status = index.Build(base, cfg);
            if (!valid) {
                REQUIRE(status != knowhere::Status::success);
                continue;
            }
            REQUIRE(status == knowhere::Status::success);
            auto result = index.Search(query, cfg, knowhere::BitsetView{});
            REQUIRE(result.has_value());
            for (int i = 0; i < 40; ++i) {
                REQUIRE(result.value()->GetIds()[i] >= 0);
                REQUIRE(result.value()->GetIds()[i] < 128);
                REQUIRE(std::isfinite(result.value()->GetDistance()[i]));
            }
        }
    }
}
