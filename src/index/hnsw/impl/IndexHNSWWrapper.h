// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/cppcontrib/knowhere/IndexWrapper.h>

#include <cstddef>
#include <cstdint>

#include "knowhere/feder/HNSW.h"

namespace knowhere {

// Custom parameters for IndexHNSW.
struct SearchParametersHNSWWrapper : public faiss::cppcontrib::knowhere::SearchParametersHNSW {
    // Stats will be updated if the object pointer is provided.
    faiss::cppcontrib::knowhere::HNSWStats* hnsw_stats = nullptr;
    // feder will be updated if the object pointer is provided.
    knowhere::feder::hnsw::FederResult* feder = nullptr;
    // filtering parameter
    float kAlpha = 1.0f;
    // Number of bits used to quantize a RaBitQ query. This is request-local
    // and must not be copied into IndexRaBitQ::qb, because searches may run
    // concurrently with different values.
    uint8_t rbq_bits_query = 0;

    inline ~SearchParametersHNSWWrapper() {
    }
};

// Creates a distance computer for the storage of an HNSW index. RaBitQ
// indexes use rbq_bits_query without changing the shared index object. The
// returned distance follows the native metric convention (in particular, IP
// is not negated).
faiss::DistanceComputer*
create_hnsw_native_distance_computer(const faiss::Index* index, uint8_t rbq_bits_query = 0);

// Creates the min-distance distance computer expected by HNSW traversal.
// Similarity metrics are wrapped in NegativeDistanceComputer.
faiss::DistanceComputer*
create_hnsw_traversal_distance_computer(const faiss::Index* index, uint8_t rbq_bits_query = 0);

// TODO:
// Please note that this particular searcher is int32_t based, so won't
//   work correctly for 2B+ samples. This can be easily changed, if needed.

// override a search() procedure for IndexHNSW.
struct IndexHNSWWrapper : public faiss::cppcontrib::knowhere::IndexWrapper {
    IndexHNSWWrapper(faiss::cppcontrib::knowhere::IndexHNSW* underlying_index);

    /// entry point for search
    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;

    /// entry point for range search
    void
    range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                 const faiss::SearchParameters* params) const override;
};

}  // namespace knowhere
