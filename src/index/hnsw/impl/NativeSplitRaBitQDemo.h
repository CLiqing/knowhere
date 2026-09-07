/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the MIT license in thirdparty/faiss/LICENSE.
 *
 * Diagnostic port of Faiss #5526 (d8a85956) bounded L2 traversal.
 * Graph adjacency is read from Knowhere without copying or changing the graph.
 * Deliberately limited to unfiltered L2 KNN; not a production search API.
 */
#pragma once

#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <cstdio>
#include <cstdlib>

namespace knowhere::native_split_demo {
struct Counts {
    size_t estimate = 0, refine = 0, expanded = 0, upper_full = 0, upper_expanded = 0;
};

template <class VT>
Counts search_one(const faiss::cppcontrib::knowhere::HNSW& graph,
                  faiss::RaBitQDistanceComputer& rq, VT& vt,
                  faiss::ResultHandler& res, int ef, bool relative) {
    Counts stats;
    using HC = faiss::CMax<float, int32_t>;
    int32_t nearest = graph.entry_point;
    float nearest_distance = rq(nearest);
    ++stats.upper_full;
    for (int level = graph.max_level; level >= 1; --level) {
        for (;;) {
            ++stats.upper_expanded;
            const int32_t previous = nearest;
            size_t begin, end;
            graph.neighbor_range(nearest, level, &begin, &end);
            int32_t ids[4];
            int count = 0;
            auto update = [&](int32_t id, float d) {
                if (d < nearest_distance) { nearest = id; nearest_distance = d; }
            };
            for (size_t j = begin; j < end && graph.neighbors[j] >= 0; ++j) {
                ids[count++] = graph.neighbors[j];
                ++stats.upper_full;
                if (count == 4) {
                    float d[4];
                    rq.distances_batch_4(ids[0], ids[1], ids[2], ids[3], d[0], d[1], d[2], d[3]);
                    for (int i = 0; i < 4; ++i) update(ids[i], d[i]);
                    count = 0;
                }
            }
            for (int i = 0; i < count; ++i) update(ids[i], rq(ids[i]));
            if (previous == nearest) break;
        }
    }

    faiss::MinimaxHeapT<HC> candidates(ef);
    candidates.push(nearest, nearest_distance);
    vt.reserve(ef);
    if (nearest_distance < res.threshold) res.add_result(nearest_distance, nearest);
    vt.set(nearest);
    while (candidates.size() > 0) {
        float d0;
        const int32_t node = candidates.pop_min(&d0);
        if (relative && candidates.count_below(d0) >= ef) break;
        size_t begin, end;
        graph.neighbor_range(node, 0, &begin, &end);
        size_t limit = begin;
        for (size_t j = begin; j < end; ++j) {
            if (graph.neighbors[j] < 0) break;
            vt.prefetch(graph.neighbors[j]);
            ++limit;
        }
        int32_t ids[4];
        int count = 0;
        float threshold = res.threshold;
        auto evaluate = [&] {
            for (int i = 0; i < count; ++i) {
                const auto* code = rq.codes + static_cast<size_t>(ids[i]) * rq.code_size;
                const float estimate = rq.distance_to_code_1bit(code);
                ++stats.estimate;
                const auto* factors = reinterpret_cast<const faiss::rabitq_utils::SignBitFactorsWithError*>(
                    code + (rq.d + 7) / 8);
                float distance = estimate;
                if (faiss::rabitq_utils::should_refine_candidate(
                        estimate, factors->f_error, rq.g_error, threshold, false)) {
                    distance = rq.distance_to_code_full(code);
                    ++stats.refine;
                }
                if (distance < threshold && res.add_result(distance, ids[i])) threshold = res.threshold;
                candidates.push(ids[i], distance);
            }
        };
        for (size_t j = begin; j < limit; ++j) {
            ids[count] = graph.neighbors[j];
            count += vt.set(ids[count]) ? 1 : 0;
            if (count == 4) { evaluate(); count = 0; }
        }
        if (count) evaluate();
        ++stats.expanded;
        if (!relative && stats.expanded > static_cast<size_t>(ef)) break;
    }
    return stats;
}

inline void search(const faiss::cppcontrib::knowhere::IndexHNSWRaBitQ& index,
                   faiss::idx_t n, const float* x, faiss::idx_t k, float* distances,
                   faiss::idx_t* labels, int ef, bool relative) {
    FAISS_THROW_IF_NOT(index.metric_type == faiss::METRIC_L2);
    FAISS_THROW_IF_NOT(index.rabitq_index()->rabitq.nb_bits > 1);
    auto raw = std::unique_ptr<faiss::FlatCodesDistanceComputer>(
        index.rabitq_index()->get_FlatCodesDistanceComputer());
    auto& rq = dynamic_cast<faiss::RaBitQDistanceComputer&>(*raw);
    // Same existing Faiss reusable visited-table implementation as the native
    // benchmark. No new custom thread-local cache is introduced by this demo.
    auto& vt = faiss::VisitedTable::get_reusable(index.ntotal);
    faiss::HeapBlockResultHandler<faiss::CMax<float, int64_t>> block(n, distances, labels, k);
    decltype(block)::SingleResultHandler result(block);
    std::vector<float> rotated(index.d);
    const bool counters = std::getenv("KNOWHERE_RBQ_TRACE_COUNTS") != nullptr;
    for (faiss::idx_t i = 0; i < n; ++i) {
        result.begin(i);
        index.pretransform_index()->chain[0]->apply_noalloc(1, x + i * index.d, rotated.data());
        rq.set_query(rotated.data());
        Counts stats;
        if (auto* vector = dynamic_cast<faiss::VisitedTableVector*>(&vt))
            stats = search_one(index.hnsw, rq, *vector, result, std::max<int>(ef, k), relative);
        else
            stats = search_one(index.hnsw, rq, dynamic_cast<faiss::VisitedTableSet&>(vt), result,
                               std::max<int>(ef, k), relative);
        result.end();
        vt.advance();
        if (counters) std::fprintf(stderr,
            "RBQ_COUNTS native ef=%d estimate=%zu refine=%zu expanded_total=%zu upper_full=%zu\n",
            ef, stats.estimate, stats.refine, stats.expanded + stats.upper_expanded, stats.upper_full);
    }
}
} // namespace knowhere::native_split_demo
