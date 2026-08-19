// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

TEST_CASE("HNSW TurboQuant build, search, and serialization", "[hnsw][turboquant]") {
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
