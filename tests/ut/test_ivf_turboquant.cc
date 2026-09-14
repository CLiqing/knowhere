// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include "catch2/catch_test_macros.hpp"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

TEST_CASE("IVF TurboQuant matches native Faiss with transforms and query parameters", "[ivf_turboquant]") {
    constexpr int nb = 512, nq = 4, d = 33, k = 20;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (bool mse : {false, true})
        for (const std::string metric : {"L2", "IP", "COSINE"}) {
            if (mse && metric == "L2")
                continue;
            for (int bits : (mse ? std::vector<int>{1, 2, 3, 4, 8} : std::vector<int>{2, 3, 4, 5})) {
                CAPTURE(mse, metric, bits);
                auto base = GenDataSet(nb, d, 814);
                auto query = GenDataSet(nq, d, 917);
                auto* xb = const_cast<float*>(static_cast<const float*>(base->GetTensor()));
                auto* xq = const_cast<float*>(static_cast<const float*>(query->GetTensor()));
                if (mse && metric == "IP") {
                    faiss::fvec_renorm_L2(d, nb, xb);
                    faiss::fvec_renorm_L2(d, nq, xq);
                }
                const std::vector<float> original_base(xb, xb + nb * d), original_query(xq, xq + nq * d);
                const auto* name = mse ? "IVF_TQMSE" : "IVF_TURBOQUANT";
                knowhere::Json cfg = {{"dim", d}, {"metric_type", metric}, {"nlist", 8},           {"nprobe", 8},
                                      {"k", k},   {"tq_bits", bits},       {"num_build_thread", 1}};
                auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
                REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
                REQUIRE(index.Count() == nb);
                REQUIRE(index.Size() > 0);
                REQUIRE(std::equal(original_base.begin(), original_base.end(), xb));
                knowhere::BinarySet binary;
                REQUIRE(index.Serialize(binary) == knowhere::Status::success);
                auto bytes = binary.GetByName(name);
                knowhere::MemoryIOReader reader(bytes->data.get(), bytes->size);
                std::unique_ptr<faiss::Index> reference(faiss::read_index(&reader));
                auto* chain = dynamic_cast<faiss::IndexPreTransform*>(reference.get());
                REQUIRE(chain != nullptr);
                auto* ivf = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(chain->index);
                REQUIRE(ivf != nullptr);
                REQUIRE_FALSE(ivf->by_residual);
                REQUIRE(chain->chain.size() == (metric == "COSINE" ? 2 : 1));
                auto loaded = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
                REQUIRE(loaded.Deserialize(binary, cfg) == knowhere::Status::success);
                for (int qb : (mse ? std::vector<int>{0} : std::vector<int>{0, 4, 8})) {
                    cfg["tq_query_bits"] = qb;
                    cfg["tq_int_qjl"] = qb != 0;
                    for (int nprobe : {1, 8})
                        for (int masked : {0, 480, 512}) {
                            CAPTURE(qb, nprobe, masked);
                            cfg["nprobe"] = nprobe;
                            std::vector<uint8_t> mask((nb + 7) / 8, 0);
                            for (int i = 0; i < masked; ++i) mask[i / 8] |= 1u << (i % 8);
                            knowhere::BitsetView bitset(mask.data(), nb);
                            knowhere::BitsetViewIDSelector selector(bitset);
                            faiss::IVFSQTurboQSearchParameters params;
                            params.nprobe = nprobe;
                            params.qb = qb;
                            params.int_qjl = qb != 0;
                            params.sel = &selector;
                            std::vector<float> distances(nq * k);
                            std::vector<faiss::idx_t> ids(nq * k);
                            // Match Knowhere's per-query transform shape (GEMV vs
                            // batched GEMM can round differently before encoding qb).
                            for (int q = 0; q < nq; ++q) {
                                reference->search(1, xq + q * d, k, distances.data() + q * k, ids.data() + q * k,
                                                  &params);
                            }
                            auto result = loaded.Search(query, cfg, bitset);
                            REQUIRE(result.has_value());
                            for (int i = 0; i < nq * k; ++i) {
                                REQUIRE(result.value()->GetIds()[i] == ids[i]);
                                REQUIRE(result.value()->GetDistance()[i] == distances[i]);
                                if (ids[i] >= 0)
                                    REQUIRE(ids[i] >= masked);
                            }
                        }
                }
                REQUIRE(std::equal(original_query.begin(), original_query.end(), xq));
                // Truncated input must fail without replacing the already loaded index.
                knowhere::BinarySet truncated;
                auto short_data = std::shared_ptr<uint8_t[]>(new uint8_t[16]);
                std::copy_n(bytes->data.get(), 16, short_data.get());
                truncated.Append(name, std::move(short_data), 16);
                REQUIRE(loaded.Deserialize(truncated, cfg) != knowhere::Status::success);
                REQUIRE(loaded.Count() == nb);
            }
        }
}

TEST_CASE("IVF TurboQuant rejects unsupported combinations", "[ivf_turboquant]") {
    auto base = GenDataSet(128, 16, 215);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (const auto* name : {"IVF_TQMSE", "IVF_TURBOQUANT"}) {
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        knowhere::Json cfg = {{"dim", 16}, {"metric_type", "COSINE"}, {"nlist", 4}, {"tq_bits", 4}};
        cfg["by_residual"] = true;
        REQUIRE(index.Build(base, cfg) == knowhere::Status::not_implemented);
        cfg["by_residual"] = false;
        cfg["refine"] = true;
        REQUIRE(index.Build(base, cfg) == knowhere::Status::not_implemented);
        cfg["refine"] = false;
        cfg["tq_bits"] = 7;
        REQUIRE(index.Build(base, cfg) == knowhere::Status::invalid_args);
        cfg["tq_bits"] = 4;
        cfg["metric_type"] = "HAMMING";
        REQUIRE(index.Build(base, cfg) == knowhere::Status::invalid_metric_type);
    }
}

TEST_CASE("IVF TurboQuant supports the fp16 and bf16 mock adapters", "[ivf_turboquant]") {
    auto check = []<typename T>() {
        auto base = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(128, 33, 149));
        auto query = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(2, 33, 151));
        const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
        for (const auto* type : {"IVF_TURBOQUANT", "IVF_TQMSE"}) {
            CAPTURE(type);
            knowhere::Json cfg = {{"dim", 33},   {"metric_type", "COSINE"}, {"nlist", 4}, {"nprobe", 4}, {"k", 10},
                                  {"tq_bits", 4}};
            auto index = knowhere::IndexFactory::Instance().Create<T>(type, version).value();
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            auto result = index.Search(query, cfg, nullptr);
            REQUIRE(result.has_value());
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            auto loaded = knowhere::IndexFactory::Instance().Create<T>(type, version).value();
            REQUIRE(loaded.Deserialize(binary, cfg) == knowhere::Status::success);
            auto roundtrip = loaded.Search(query, cfg, nullptr);
            REQUIRE(roundtrip.has_value());
            for (int i = 0; i < 20; ++i) {
                REQUIRE(std::isfinite(result.value()->GetDistance()[i]));
                REQUIRE(result.value()->GetIds()[i] == roundtrip.value()->GetIds()[i]);
                REQUIRE(result.value()->GetDistance()[i] == roundtrip.value()->GetDistance()[i]);
            }
        }
    };
    check.template operator()<knowhere::fp16>();
    check.template operator()<knowhere::bf16>();
}
