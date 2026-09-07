#include <algorithm>
#include <cmath>
#include <vector>
#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"
#include "knowhere/bitsetview.h"
#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/cppcontrib/knowhere/impl/StagedDistanceComputer.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/distances.h>
#include "catch2/catch_approx.hpp"
#include "index/hnsw/impl/NativeSplitRaBitQDemo.h"
#include <faiss/utils/rabitq_simd.h>

TEST_CASE("Split AVX512 full scorer matches scalar for multi-bit tails", "[hnsw_split_native]") {
#if defined(__GNUC__) && defined(__x86_64__)
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl") ||
        !__builtin_cpu_supports("bmi2")) return;
    for (size_t d : {8, 16, 23, 24, 31, 65, 200, 768, 1536}) {
        for (size_t ex : {2, 3, 4, 5, 6, 7}) {
            for (int seed = 1; seed <= 3; ++seed) {
                CAPTURE(d, ex, seed);
                std::vector<uint8_t> signs((d+7)/8), extra((d*ex+7)/8+32);
                std::vector<float> query(d);
                for (size_t i=0; i<signs.size(); ++i) signs[i]=(i*73+seed*19)%256;
                for (size_t i=0; i<extra.size(); ++i) extra[i]=(i*131+seed*37)%256;
                for (size_t i=0; i<d; ++i) query[i]=std::sin(float(i)*.37f+seed);
                const float cb=-float(1u<<ex)+.5f;
                const float ref=faiss::rabitq::multibit::compute_inner_product<faiss::SIMDLevel::NONE>(
                    signs.data(),extra.data(),query.data(),d,ex,cb);
                const float actual=faiss::rabitq::multibit::compute_inner_product<faiss::SIMDLevel::AVX512>(
                    signs.data(),extra.data(),query.data(),d,ex,cb);
                REQUIRE(actual == Catch::Approx(ref).epsilon(1e-5).margin(1e-3));
            }
        }
    }
#endif
}

TEST_CASE("Native split traversal retains all results when k covers the graph", "[hnsw_split_native]") {
    namespace fk = faiss::cppcontrib::knowhere;
    constexpr int n = 128, dim = 65;
    auto base = GenDataSet(n, dim, 121);
    auto queries = GenDataSet(4, dim, 122);
    const auto* x = static_cast<const float*>(base->GetTensor());
    fk::IndexHNSWFlat fp32(dim, 16, faiss::METRIC_L2);
    fp32.add(n, x);
    auto* rq = new faiss::IndexRaBitQ(dim, faiss::METRIC_L2, 8);
    rq->qb = 4;
    faiss::IndexPreTransform storage(new faiss::RandomRotationMatrix(dim, dim), rq);
    storage.own_fields = true;
    storage.train(n, x);
    storage.add(n, x);
    fk::IndexHNSWRaBitQ graph;
    graph.d = dim; graph.ntotal = n; graph.metric_type = faiss::METRIC_L2;
    graph.storage = &storage; graph.own_fields = false;
    graph.hnsw = std::move(fp32.hnsw);
    std::unique_ptr<faiss::DistanceComputer> full(storage.get_distance_computer());
    std::vector<float> distances(4 * n);
    std::vector<faiss::idx_t> labels(4 * n);
    knowhere::native_split_demo::search(graph, 4,
        static_cast<const float*>(queries->GetTensor()), n, distances.data(), labels.data(), n, true);
    for (int q = 0; q < 4; ++q) {
        full->set_query(static_cast<const float*>(queries->GetTensor()) + q * dim);
        std::vector<std::pair<float, faiss::idx_t>> expected;
        for (int i = 0; i < n; ++i) expected.emplace_back((*full)(i), i);
        std::sort(expected.begin(), expected.end());
        for (int i = 0; i < n; ++i) {
            REQUIRE(labels[q * n + i] == expected[i].second);
            REQUIRE(distances[q * n + i] == Catch::Approx(expected[i].first).margin(1e-5));
        }
    }
}

TEST_CASE("Split staged distances preserve metric and cosine threshold semantics", "[hnsw_split_rabitq]") {
    namespace fk = faiss::cppcontrib::knowhere;
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        const bool cosine = std::string(metric)=="COSINE";
        const bool similarity = std::string(metric)!="L2";
        for (int qb : {0,4}) {
            auto base=GenDataSet(128,65,4201);
            auto query=GenDataSet(4,65,4202);
            const auto* x=static_cast<const float*>(base->GetTensor());
            auto* rr=new faiss::RandomRotationMatrix(65,65);
            auto* rq=new faiss::IndexRaBitQ(65, similarity ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2,8);
            rq->qb=qb;
            std::unique_ptr<faiss::IndexPreTransform> storage(cosine
                ? new fk::IndexPreTransformRaBitQCosine(rr,rq) : new faiss::IndexPreTransform(rr,rq));
            storage->own_fields=true;
            storage->train(128,x); storage->add(128,x);
            std::unique_ptr<fk::IndexHNSWRaBitQ> graph(cosine
                ? new fk::IndexHNSWRaBitQCosine() : new fk::IndexHNSWRaBitQ());
            graph->d=65; graph->metric_type=rq->metric_type; graph->storage=storage.get(); graph->own_fields=false;
            std::unique_ptr<faiss::DistanceComputer> staged_owner(graph->get_staged_distance_computer());
            auto* staged=dynamic_cast<fk::StagedDistanceComputer*>(staged_owner.get());
            REQUIRE(staged!=nullptr);
            std::unique_ptr<faiss::DistanceComputer> full(storage->get_distance_computer());
            std::unique_ptr<faiss::FlatCodesDistanceComputer> raw_owner(rq->get_FlatCodesDistanceComputer());
            auto* raw=dynamic_cast<faiss::RaBitQDistanceComputer*>(raw_owner.get());
            REQUIRE(raw!=nullptr);
            std::vector<float> rotated(65);
            for (int q=0;q<4;++q) {
                const auto* v=static_cast<const float*>(query->GetTensor())+q*65;
                staged->set_query(v); full->set_query(v);
                rr->apply_noalloc(1,v,rotated.data()); raw->set_query(rotated.data());
                for(int i=0;i<128;++i) {
                    const float expected=(similarity ? -1 : 1)*(*full)(i);
                    REQUIRE((*staged)(i)==Catch::Approx(expected).margin(1e-5));
                    REQUIRE(staged->evaluate(i,std::numeric_limits<float>::infinity())==Catch::Approx(expected).margin(1e-5));
                    const float scale=cosine
                        ? dynamic_cast<fk::IndexPreTransformRaBitQCosine*>(storage.get())->get_inverse_l2_norms()[i]
                          / std::sqrt(faiss::fvec_norm_L2sqr(v,65)) : 1;
                    const float estimate=raw->distance_to_code_1bit(raw->codes+i*raw->code_size)
                        * scale * (similarity ? -1 : 1);
                    REQUIRE(staged->evaluate(i,-std::numeric_limits<float>::infinity())==Catch::Approx(estimate).margin(1e-5));
                }
                REQUIRE(staged->estimate_count==256);
                REQUIRE(staged->refine_count==128);
            }
        }
    }
}

TEST_CASE("Split HNSW RaBitQ metrics and serialized search", "[hnsw_split_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (int qb : {0, 4}) {
            CAPTURE(metric, qb);
            auto base = GenDataSet(1024, 65, 3101);
            auto query = GenDataSet(16, 65, 3102);
            const auto* data = static_cast<const float*>(base->GetTensor());
            std::vector<float> original(data, data + 1024 * 65);
            knowhere::Json config = {{"dim",65}, {"metric_type",metric}, {"k",20},
                {"M",16}, {"efConstruction",100}, {"ef",200}, {"rbq_bits",8},
                {"rbq_bits_query",qb}};
            auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_HNSW_RABITQ, version).value();
            REQUIRE(index.Build(base, config) == knowhere::Status::success);
            REQUIRE(std::equal(original.begin(), original.end(), data));
            auto before = index.Search(query, config, nullptr);
            REQUIRE(before.has_value());
            for (int excluded : {128, 960}) {
                std::vector<uint8_t> mask(128,0);
                for (int i=0;i<excluded;++i) mask[i/8] |= uint8_t(1u << (i%8));
                knowhere::BitsetView filter(mask.data(),1024);
                auto filtered = index.Search(query,config,filter);
                REQUIRE(filtered.has_value());
                for (int i=0;i<320;++i) {
                    REQUIRE(filtered.value()->GetIds()[i]>=excluded);
                    REQUIRE(filtered.value()->GetIds()[i]<1024);
                }
            }
            for (int q=0; q<16; ++q) for (int j=0; j<20; ++j) {
                int i=q*20+j;
                REQUIRE(before.value()->GetIds()[i] >= 0);
                REQUIRE(before.value()->GetIds()[i] < 1024);
                REQUIRE(std::isfinite(before.value()->GetDistance()[i]));
                if (j) {
                    if (std::string(metric)=="L2") REQUIRE(before.value()->GetDistance()[i-1] <= before.value()->GetDistance()[i]);
                    else REQUIRE(before.value()->GetDistance()[i-1] >= before.value()->GetDistance()[i]);
                }
            }
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            auto loaded = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_HNSW_RABITQ, version).value();
            REQUIRE(loaded.Deserialize(binary, config) == knowhere::Status::success);
            auto after = loaded.Search(query, config, nullptr);
            REQUIRE(after.has_value());
            for (int i=0;i<320;++i) {
                REQUIRE(before.value()->GetIds()[i] == after.value()->GetIds()[i]);
                REQUIRE(before.value()->GetDistance()[i] == after.value()->GetDistance()[i]);
            }
        }
    }
}
