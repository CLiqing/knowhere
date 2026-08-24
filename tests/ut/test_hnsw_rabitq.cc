// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "faiss/IndexRaBitQ.h"
#include "faiss/impl/RaBitQUtils.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace {

constexpr int64_t kNb = 256;
constexpr int64_t kNq = 4;
constexpr int64_t kTopk = 8;

knowhere::Json
MakeHnswRaBitQConfig(int64_t dim, const std::string& metric, int rbq_bits = 1) {
    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::ROWS] = kNb;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = kTopk;
    json[knowhere::indexparam::HNSW_M] = 12;
    json[knowhere::indexparam::EFCONSTRUCTION] = 64;
    json[knowhere::indexparam::EF] = 64;
    json[knowhere::indexparam::RABITQ_BITS] = rbq_bits;
    return json;
}

bool
SerializedIndexContainsFourcc(const knowhere::BinarySet& binary_set, const std::string& name, const char* fourcc) {
    const auto binary = binary_set.GetByName(name);
    if (binary == nullptr) {
        return false;
    }
    const auto* begin = binary->data.get();
    const auto* end = begin + binary->size;
    return std::search(begin, end, fourcc, fourcc + 4) != end;
}

void
CheckValidKnnResult(const knowhere::DataSet& result, int64_t nb, int64_t nq, int64_t topk) {
    REQUIRE(result.GetRows() == nq);
    REQUIRE(result.GetDim() == topk);
    const auto* ids = result.GetIds();
    const auto* distances = result.GetDistance();
    for (int64_t i = 0; i < nq * topk; ++i) {
        REQUIRE(ids[i] >= 0);
        REQUIRE(ids[i] < nb);
        REQUIRE(std::isfinite(distances[i]));
    }
}

void
CheckKnnOrder(const knowhere::DataSet& result, const std::string& metric) {
    const auto rows = result.GetRows();
    const auto topk = result.GetDim();
    const auto* distances = result.GetDistance();
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 1; j < topk; ++j) {
            const auto previous = distances[i * topk + j - 1];
            const auto current = distances[i * topk + j];
            if (knowhere::IsMetricType(metric, knowhere::metric::L2)) {
                REQUIRE(previous <= current);
            } else {
                REQUIRE(previous >= current);
            }
        }
    }
}

template <typename DataType>
void
CheckTypedBuildAndSearch(int64_t dim) {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance().Create<DataType>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version);
    REQUIRE(index.has_value());

    const auto train = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(kNb, dim, 101));
    const auto query = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(kNq, dim, 102));
    auto json = MakeHnswRaBitQConfig(dim, knowhere::metric::L2);

    REQUIRE(index.value().Build(train, json) == knowhere::Status::success);
    const auto result = index.value().Search(query, json, nullptr);
    REQUIRE(result.has_value());
    CheckValidKnnResult(*result.value(), kNb, kNq, kTopk);
}

struct SearchSnapshot {
    std::vector<int64_t> ids;
    std::vector<float> distances;
};

SearchSnapshot
TakeSnapshot(const knowhere::DataSet& result) {
    const auto count = result.GetRows() * result.GetDim();
    return {std::vector<int64_t>(result.GetIds(), result.GetIds() + count),
            std::vector<float>(result.GetDistance(), result.GetDistance() + count)};
}

}  // namespace

TEST_CASE("HNSW_RABITQ supports the public type and data-type boundary", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto type = knowhere::IndexEnum::INDEX_HNSW_RABITQ;
    auto& factory = knowhere::IndexFactory::Instance();

    REQUIRE(factory.Create<knowhere::fp32>(type, version).has_value());
    REQUIRE(factory.Create<knowhere::fp16>(type, version).has_value());
    REQUIRE(factory.Create<knowhere::bf16>(type, version).has_value());
    REQUIRE_FALSE(factory.Create<knowhere::int8>(type, version).has_value());
    REQUIRE_FALSE(factory.Create<knowhere::bin1>(type, version).has_value());

    REQUIRE(factory.FeatureCheck(type, knowhere::feature::FLOAT32));
    REQUIRE(factory.FeatureCheck(type, knowhere::feature::FP16));
    REQUIRE(factory.FeatureCheck(type, knowhere::feature::BF16));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::INT8));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::BINARY));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::MMAP));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::MV));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::EMB_LIST));

    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_FLOAT));
    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_FLOAT16));
    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_BFLOAT16));
    REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_INT8));
    REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_BINARY));
    REQUIRE_FALSE(knowhere::KnowhereCheck::SupportMmapIndexTypeCheck(type));
    REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_FLOAT, true));

    CheckTypedBuildAndSearch<knowhere::fp32>(13);
    CheckTypedBuildAndSearch<knowhere::fp16>(13);
    CheckTypedBuildAndSearch<knowhere::bf16>(13);
}

TEST_CASE("HNSW_RABITQ validates metrics and bit widths", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto train = GenDataSet(kNb, 32, 201);
    const auto query = GenDataSet(kNq, 32, 202);

    for (const int rbq_bits : {0, 10}) {
        CAPTURE(rbq_bits);
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        const auto json = MakeHnswRaBitQConfig(32, knowhere::metric::L2, rbq_bits);
        REQUIRE(index.Build(train, json) == knowhere::Status::out_of_range_in_json);
    }

    {
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        const auto json = MakeHnswRaBitQConfig(32, knowhere::metric::COSINE);
        REQUIRE(index.Build(train, json) == knowhere::Status::invalid_metric_type);
    }

    {
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        auto json = MakeHnswRaBitQConfig(32, knowhere::metric::L2);
        REQUIRE(index.Build(train, json) == knowhere::Status::success);

        json[knowhere::indexparam::RABITQ_QUERY_BITS] = 9;
        const auto result = index.Search(query, json, nullptr);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == knowhere::Status::out_of_range_in_json);
    }

    {
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        auto json = MakeHnswRaBitQConfig(32, knowhere::metric::L2, 4);
        REQUIRE(index.Build(train, json) == knowhere::Status::success);

        json[knowhere::indexparam::RABITQ_QUERY_BITS] = 1;
        const auto result = index.Search(query, json, nullptr);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == knowhere::Status::invalid_args);
    }
}

TEST_CASE("HNSW_RABITQ searches, ranges, iterates, and round-trips", "[hnsw_rabitq]") {
    struct Scenario {
        const char* metric;
        int64_t dim;
        int rbq_bits;
        int rbq_bits_query;
    };
    const std::vector<Scenario> scenarios = {
        {knowhere::metric::L2, 127, 1, 8},
        {knowhere::metric::IP, 129, 1, 8},
        {knowhere::metric::L2, 129, 2, 0},
        {knowhere::metric::L2, 128, 8, 0},
        {knowhere::metric::IP, 13, 9, 0},
    };
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    for (const auto& scenario : scenarios) {
        CAPTURE(scenario.metric, scenario.dim, scenario.rbq_bits, scenario.rbq_bits_query);
        const auto train = GenDataSet(kNb, scenario.dim, 301 + scenario.dim + scenario.rbq_bits);
        const auto query = GenDataSet(kNq, scenario.dim, 401 + scenario.dim + scenario.rbq_bits);
        auto json = MakeHnswRaBitQConfig(scenario.dim, scenario.metric, scenario.rbq_bits);
        json[knowhere::indexparam::RABITQ_QUERY_BITS] = scenario.rbq_bits_query;

        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        REQUIRE(index.Build(train, json) == knowhere::Status::success);
        REQUIRE(index.Count() == kNb);

        const auto before = index.Search(query, json, nullptr);
        REQUIRE(before.has_value());
        CheckValidKnnResult(*before.value(), kNb, kNq, kTopk);
        CheckKnnOrder(*before.value(), scenario.metric);

        auto range_json = json;
        if (knowhere::IsMetricType(scenario.metric, knowhere::metric::L2)) {
            range_json[knowhere::meta::RADIUS] = std::numeric_limits<float>::max();
            range_json[knowhere::meta::RANGE_FILTER] = 0.0f;
        } else {
            range_json[knowhere::meta::RADIUS] = -std::numeric_limits<float>::max();
            range_json[knowhere::meta::RANGE_FILTER] = std::numeric_limits<float>::max();
        }
        range_json[knowhere::meta::RANGE_SEARCH_K] = 32;
        const auto range_result = index.RangeSearch(query, range_json, nullptr);
        REQUIRE(range_result.has_value());
        REQUIRE(range_result.value()->GetLims()[kNq] > 0);

        const auto iterators = index.AnnIterator(query, json, nullptr, false);
        REQUIRE(iterators.has_value());
        REQUIRE(iterators.value().size() == kNq);
        for (auto& iterator : iterators.value()) {
            REQUIRE(iterator->HasNext().value());
            const auto next = iterator->Next();
            REQUIRE(next.has_value());
            REQUIRE(next.value().first >= 0);
            REQUIRE(next.value().first < kNb);
            REQUIRE(std::isfinite(next.value().second));
        }

        if (scenario.rbq_bits_query == 0) {
            const auto* labels = before.value()->GetIds();
            const auto one_query = knowhere::GenDataSet(1, scenario.dim, query->GetTensor());
            const auto distances = index.CalcDistByIDs(one_query, nullptr, labels, kTopk, false);
            REQUIRE(distances.has_value());
            for (int64_t i = 0; i < kTopk; ++i) {
                REQUIRE(distances.value()->GetDistance()[i] ==
                        Catch::Approx(before.value()->GetDistance()[i]).epsilon(1e-5));
            }
        }

        knowhere::BinarySet binary_set;
        REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
        REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IHNr"));
        const char* storage_fourcc = scenario.rbq_bits == 1 ? "Ixrq" : (scenario.rbq_bits == 8 ? "Ixrb" : "Ixrr");
        REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), storage_fourcc));

        auto loaded = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                          .value();
        REQUIRE(loaded.Deserialize(binary_set, json) == knowhere::Status::success);
        const auto after = loaded.Search(query, json, nullptr);
        REQUIRE(after.has_value());
        const auto before_snapshot = TakeSnapshot(*before.value());
        const auto after_snapshot = TakeSnapshot(*after.value());
        REQUIRE(after_snapshot.ids == before_snapshot.ids);
        REQUIRE(after_snapshot.distances == before_snapshot.distances);

        REQUIRE(index.Add(GenDataSet(8, scenario.dim, 501), json) == knowhere::Status::not_implemented);
    }
}

TEST_CASE("RaBitQ8 byte layout preserves packed distances", "[hnsw_rabitq]") {
    constexpr int64_t dim = 128;
    constexpr int64_t nb = 64;
    constexpr int64_t nq = 2;
    const auto base = GenDataSet(nb, dim, 1701);
    const auto query = GenDataSet(nq, dim, 1702);
    const auto* base_data = static_cast<const float*>(base->GetTensor());
    const auto* query_data = static_cast<const float*>(query->GetTensor());

    for (const auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        faiss::IndexRaBitQ packed(dim, metric, 8, false);
        faiss::IndexRaBitQ bytes(dim, metric, 8, true);
        packed.qb = 0;
        bytes.qb = 0;
        packed.train(nb, base_data);
        bytes.train(nb, base_data);
        packed.add(nb, base_data);
        bytes.add(nb, base_data);

        REQUIRE(packed.code_size == bytes.code_size);
        for (int64_t i = 0; i < nb; ++i) {
            const uint8_t* packed_code = packed.codes.data() + i * packed.code_size;
            const uint8_t* byte_code = bytes.codes.data() + i * bytes.code_size;
            const uint8_t* extra_code = packed_code + (dim + 7) / 8 +
                    sizeof(faiss::rabitq_utils::SignBitFactorsWithError);
            for (int64_t j = 0; j < dim; ++j) {
                const uint8_t sign = (packed_code[j / 8] & (1u << (j % 8))) != 0 ? 0x80 : 0;
                const uint8_t extra = static_cast<uint8_t>(
                        faiss::rabitq_utils::extract_code_inline(extra_code, j, 7));
                REQUIRE(byte_code[j] == static_cast<uint8_t>(sign | extra));
            }
        }

        std::unique_ptr<faiss::DistanceComputer> packed_dc(packed.get_distance_computer());
        std::unique_ptr<faiss::DistanceComputer> byte_dc(bytes.get_distance_computer());
        for (int64_t q = 0; q < nq; ++q) {
            packed_dc->set_query(query_data + q * dim);
            byte_dc->set_query(query_data + q * dim);
            for (int64_t i = 0; i < nb; ++i) {
                REQUIRE((*byte_dc)(i) == Catch::Approx((*packed_dc)(i)).epsilon(2e-5));
            }
        }
    }
}

TEST_CASE("HNSW_RABITQ request-local query bits cover BF and concurrent search", "[hnsw_rabitq]") {
    constexpr int64_t dim = 64;
    constexpr int64_t nb = 384;
    constexpr int64_t nq = 3;
    constexpr int64_t topk = 10;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto train = GenDataSet(nb, dim, 601);
    const auto query = GenDataSet(nq, dim, 602);

    auto json = MakeHnswRaBitQConfig(dim, knowhere::metric::IP);
    json[knowhere::meta::ROWS] = nb;
    json[knowhere::meta::TOPK] = topk;
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                     .value();
    REQUIRE(index.Build(train, json) == knowhere::Status::success);

    auto json_qb0 = json;
    json_qb0[knowhere::indexparam::RABITQ_QUERY_BITS] = 0;
    auto json_qb8 = json;
    json_qb8[knowhere::indexparam::RABITQ_QUERY_BITS] = 8;

    const auto result_qb0 = index.Search(query, json_qb0, nullptr);
    const auto result_qb8 = index.Search(query, json_qb8, nullptr);
    REQUIRE(result_qb0.has_value());
    REQUIRE(result_qb8.has_value());
    const auto expected_qb0 = TakeSnapshot(*result_qb0.value());
    const auto expected_qb8 = TakeSnapshot(*result_qb8.value());

    std::atomic<bool> consistent = true;
    std::vector<std::thread> threads;
    for (int thread_id = 0; thread_id < 8; ++thread_id) {
        threads.emplace_back([&, thread_id] {
            const auto& thread_json = (thread_id % 2 == 0) ? json_qb0 : json_qb8;
            const auto& expected = (thread_id % 2 == 0) ? expected_qb0 : expected_qb8;
            for (int iteration = 0; iteration < 8 && consistent.load(); ++iteration) {
                const auto result = index.Search(query, thread_json, nullptr);
                if (!result.has_value()) {
                    consistent.store(false);
                    break;
                }
                const auto actual = TakeSnapshot(*result.value());
                if (actual.ids != expected.ids || actual.distances != expected.distances) {
                    consistent.store(false);
                    break;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(consistent.load());

    const int64_t kept = 16;
    std::vector<uint8_t> bitset_data((nb + 7) / 8, 0);
    for (int64_t i = 0; i < nb - kept; ++i) {
        bitset_data[i >> 3] |= static_cast<uint8_t>(1U << (i & 7));
    }
    const knowhere::BitsetView bitset(bitset_data.data(), nb, nb - kept);

    auto bf_json = json_qb8;
    bf_json[knowhere::meta::TOPK] = 8;
    const auto bf_result = index.Search(query, bf_json, bitset);
    REQUIRE(bf_result.has_value());
    CheckValidKnnResult(*bf_result.value(), nb, nq, 8);
    CheckKnnOrder(*bf_result.value(), knowhere::metric::IP);
    for (int64_t i = 0; i < nq * 8; ++i) {
        REQUIRE(bf_result.value()->GetIds()[i] >= nb - kept);
    }

    std::vector<int64_t> kept_ids(kept);
    for (int64_t i = 0; i < kept; ++i) {
        kept_ids[i] = nb - kept + i;
    }
    const auto one_query = knowhere::GenDataSet(1, dim, query->GetTensor());
    const auto kept_distances = index.CalcDistByIDs(one_query, nullptr, kept_ids.data(), kept_ids.size(), false);
    REQUIRE(kept_distances.has_value());
    std::vector<std::pair<float, int64_t>> expected_ip;
    expected_ip.reserve(kept);
    for (int64_t i = 0; i < kept; ++i) {
        expected_ip.emplace_back(kept_distances.value()->GetDistance()[i], kept_ids[i]);
    }
    std::sort(expected_ip.begin(), expected_ip.end(), [](const auto& left, const auto& right) {
        return left.first != right.first ? left.first > right.first : left.second < right.second;
    });

    auto bf_json_qb0 = bf_json;
    bf_json_qb0[knowhere::indexparam::RABITQ_QUERY_BITS] = 0;
    const auto bf_result_qb0 = index.Search(one_query, bf_json_qb0, bitset);
    REQUIRE(bf_result_qb0.has_value());
    for (int64_t i = 0; i < 8; ++i) {
        REQUIRE(bf_result_qb0.value()->GetIds()[i] == expected_ip[i].second);
        REQUIRE(bf_result_qb0.value()->GetDistance()[i] == Catch::Approx(expected_ip[i].first).epsilon(1e-6));
    }

    auto range_json = bf_json;
    range_json[knowhere::meta::RADIUS] = -std::numeric_limits<float>::max();
    range_json[knowhere::meta::RANGE_FILTER] = std::numeric_limits<float>::max();
    range_json[knowhere::meta::RANGE_SEARCH_K] = -1;
    const auto range_result = index.RangeSearch(query, range_json, bitset);
    REQUIRE(range_result.has_value());
    REQUIRE(range_result.value()->GetLims()[nq] > 0);
    for (size_t i = 0; i < range_result.value()->GetLims()[nq]; ++i) {
        REQUIRE(range_result.value()->GetIds()[i] >= nb - kept);
    }
}

TEST_CASE("HNSW_RABITQ supports optional refinement", "[hnsw_rabitq]") {
    constexpr int64_t dim = 32;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto train = GenDataSet(kNb, dim, 701);
    const auto query = GenDataSet(kNq, dim, 702);
    auto json = MakeHnswRaBitQConfig(dim, knowhere::metric::L2);
    json[knowhere::indexparam::REFINE] = true;
    json[knowhere::indexparam::REFINE_TYPE] = "FLAT";
    json[knowhere::indexparam::REFINE_K] = 2.0f;

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                     .value();
    REQUIRE(index.Build(train, json) == knowhere::Status::success);
    REQUIRE(index.IsIndexRefineEnabled());
    REQUIRE(index.HasRawData(knowhere::metric::L2));

    const auto result = index.Search(query, json, nullptr);
    REQUIRE(result.has_value());
    CheckValidKnnResult(*result.value(), kNb, kNq, kTopk);

    knowhere::BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
    REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IHNr"));
    REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "Ixrq"));

    auto loaded = knowhere::IndexFactory::Instance()
                      .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                      .value();
    REQUIRE(loaded.Deserialize(binary_set, json) == knowhere::Status::success);
    const auto loaded_result = loaded.Search(query, json, nullptr);
    REQUIRE(loaded_result.has_value());
    REQUIRE(TakeSnapshot(*loaded_result.value()).ids == TakeSnapshot(*result.value()).ids);
}
