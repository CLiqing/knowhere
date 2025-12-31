// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <faiss/IndexSQ8Uniform.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <faiss/FaissHook.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/prefetch.h>
#include <knowhere/utils.h>

namespace faiss {

//////////////////////////////////////////////////////////////////////////////////
// SQ8UniformCosineDistanceComputer implementation
//////////////////////////////////////////////////////////////////////////////////

SQ8UniformCosineDistanceComputer::SQ8UniformCosineDistanceComputer(
        const int d_,
        std::unique_ptr<DistanceComputer>&& basedis_)
        : basedis(std::move(basedis_)), d(d_) {}

void SQ8UniformCosineDistanceComputer::set_query(const float* x) {
    // For COSINE metric, normalize query vector before computing distances
    // At this point, data has already been converted to float by knowhere layer
    query_storage.resize(d);
    std::copy_n(x, d, query_storage.begin());

    // Normalize using knowhere's function
    knowhere::NormalizeVec<float>(query_storage.data(), d);

    // Set the normalized query to base distance computer
    basedis->set_query(query_storage.data());
}

float SQ8UniformCosineDistanceComputer::operator()(idx_t i) {
    float l2_sqr_dis = (*basedis)(i);
    return 1.0f - 0.5f * l2_sqr_dis;
}

void SQ8UniformCosineDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    basedis->distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);

    dis0 = 1.0f - 0.5f * dis0;
    dis1 = 1.0f - 0.5f * dis1;
    dis2 = 1.0f - 0.5f * dis2;
    dis3 = 1.0f - 0.5f * dis3;
}

float SQ8UniformCosineDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    float l2_sqr_dis = basedis->symmetric_dis(i, j);
    return 1.0f - 0.5f * l2_sqr_dis;
}

//////////////////////////////////////////////////////////////////////////////////
// WithSQ8UniformNormIPDistanceComputer implementation
//////////////////////////////////////////////////////////////////////////////////

WithSQ8UniformNormIPDistanceComputer::WithSQ8UniformNormIPDistanceComputer(
        const float* l2_norms_sqr_,
        const int d_,
        std::unique_ptr<DistanceComputer>&& basedis_)
        : basedis(std::move(basedis_)), l2_norms_sqr(l2_norms_sqr_), d(d_) {}

void WithSQ8UniformNormIPDistanceComputer::set_query(const float* x) {
    if (x != nullptr) {
        // For IP: compute query norm squared for distance conversion
        query_norm_sqr = faiss::fvec_norm_L2sqr(x, d);
        if (query_norm_sqr <= 0) {
            query_norm_sqr = 1.0f;
        }
        basedis->set_query(x);
    } else {
        query_norm_sqr = 0;
        basedis->set_query(nullptr);
    }
}

float WithSQ8UniformNormIPDistanceComputer::operator()(idx_t i) {
    float l2_sqr_dis = (*basedis)(i);
    prefetch_L2(l2_norms_sqr + i);
    const float base_norm_sqr = l2_norms_sqr[i];
    return 0.5f * (query_norm_sqr + base_norm_sqr - l2_sqr_dis);
}

void WithSQ8UniformNormIPDistanceComputer::distances_batch_4(
        const idx_t idx0,
        const idx_t idx1,
        const idx_t idx2,
        const idx_t idx3,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    basedis->distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);

    prefetch_L2(l2_norms_sqr + idx0);
    prefetch_L2(l2_norms_sqr + idx1);
    prefetch_L2(l2_norms_sqr + idx2);
    prefetch_L2(l2_norms_sqr + idx3);

    const float base_norm_sqr0 = l2_norms_sqr[idx0];
    const float base_norm_sqr1 = l2_norms_sqr[idx1];
    const float base_norm_sqr2 = l2_norms_sqr[idx2];
    const float base_norm_sqr3 = l2_norms_sqr[idx3];

    dis0 = 0.5f * (query_norm_sqr + base_norm_sqr0 - dis0);
    dis1 = 0.5f * (query_norm_sqr + base_norm_sqr1 - dis1);
    dis2 = 0.5f * (query_norm_sqr + base_norm_sqr2 - dis2);
    dis3 = 0.5f * (query_norm_sqr + base_norm_sqr3 - dis3);
}

float WithSQ8UniformNormIPDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    float l2_sqr_dis = basedis->symmetric_dis(i, j);

    prefetch_L2(l2_norms_sqr + i);
    prefetch_L2(l2_norms_sqr + j);

    const float norm_i_sqr = l2_norms_sqr[i];
    const float norm_j_sqr = l2_norms_sqr[j];

    return 0.5f * (norm_i_sqr + norm_j_sqr - l2_sqr_dis);
}

//////////////////////////////////////////////////////////////////////////////////
// IndexScalarQuantizer8bitUniformCosine implementation
//////////////////////////////////////////////////////////////////////////////////

IndexScalarQuantizer8bitUniformCosine::IndexScalarQuantizer8bitUniformCosine(
        int d)
        : IndexScalarQuantizer(
                  d,
                  ScalarQuantizer::QT_8bit_uniform,
                  METRIC_INNER_PRODUCT) {
    is_cosine = true;

    sq.rangestat = ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.01;
}

IndexScalarQuantizer8bitUniformCosine::IndexScalarQuantizer8bitUniformCosine()
        : IndexScalarQuantizer() {
    metric_type = METRIC_INNER_PRODUCT;
    is_cosine = true;

    sq.rangestat = ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.01;
}

void IndexScalarQuantizer8bitUniformCosine::train(idx_t n, const float* x) {
    // For COSINE metric, normalize vectors before training
    // Use knowhere's CopyAndNormalizeVecs to avoid modifying input data
    auto normalized_data = knowhere::CopyAndNormalizeVecs<float>(x, n, d);

    // Train on normalized data
    sq.train(n, normalized_data.get());
    is_trained = true;
}

void IndexScalarQuantizer8bitUniformCosine::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);

    // For COSINE metric, normalize vectors before adding
    // Use knowhere's CopyAndNormalizeVecs to avoid modifying input data
    auto normalized_data = knowhere::CopyAndNormalizeVecs<float>(x, n, d);

    // Add normalized data
    IndexScalarQuantizer::add(n, normalized_data.get());

    // Calculate and store inverse L2 norms from ORIGINAL vectors (not
    // normalized) This is needed for refine to work correctly with COSINE
    // metric
    const size_t current_size = inverse_l2_norms.size();
    inverse_l2_norms.resize(current_size + n);
    for (idx_t i = 0; i < n; i++) {
        const float l2sqr_norm = fvec_norm_L2sqr(x + i * d, d);
        const float inverse_l2_norm =
                (l2sqr_norm == 0.0f) ? 1.0f : (1.0f / sqrtf(l2sqr_norm));
        inverse_l2_norms[i + current_size] = inverse_l2_norm;
    }
}

DistanceComputer* IndexScalarQuantizer8bitUniformCosine::get_distance_computer()
        const {
    std::unique_ptr<DistanceComputer> base_dc(
            IndexScalarQuantizer::get_distance_computer());

    return new SQ8UniformCosineDistanceComputer(d, std::move(base_dc));
}

const float* IndexScalarQuantizer8bitUniformCosine::get_inverse_l2_norms()
        const {
    // Ensure cache is sized correctly
    if (inverse_l2_norms.size() != static_cast<size_t>(ntotal)) {
        inverse_l2_norms.resize(ntotal, 1.0f);
    }
    return inverse_l2_norms.data();
}

void IndexScalarQuantizer8bitUniformCosine::reset() {
    IndexScalarQuantizer::reset();
    inverse_l2_norms.clear();
}

//////////////////////////////////////////////////////////////////////////////////
// IndexScalarQuantizer8bitUniformIP implementation
//////////////////////////////////////////////////////////////////////////////////

IndexScalarQuantizer8bitUniformIP::IndexScalarQuantizer8bitUniformIP(int d)
        : IndexScalarQuantizer(
                  d,
                  ScalarQuantizer::QT_8bit_uniform,
                  METRIC_INNER_PRODUCT) {
    is_cosine = false;
}

IndexScalarQuantizer8bitUniformIP::IndexScalarQuantizer8bitUniformIP()
        : IndexScalarQuantizer() {
    metric_type = METRIC_INNER_PRODUCT;
    is_cosine = false;
}

void IndexScalarQuantizer8bitUniformIP::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    IndexScalarQuantizer::add(n, x);

    // Compute and store norms squared for IP distance computation
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        float norm_sqr = faiss::fvec_norm_L2sqr(vec, d);
        l2_norms_sqr.push_back(norm_sqr > 0 ? norm_sqr : 1.0f);
    }
}

void IndexScalarQuantizer8bitUniformIP::reset() {
    IndexScalarQuantizer::reset();
    l2_norms_sqr.clear();
}

DistanceComputer* IndexScalarQuantizer8bitUniformIP::get_distance_computer()
        const {
    std::unique_ptr<DistanceComputer> base_dc(
            IndexScalarQuantizer::get_distance_computer());

    return new WithSQ8UniformNormIPDistanceComputer(
            get_l2_norms_sqr(), d, std::move(base_dc));
}

const float* IndexScalarQuantizer8bitUniformIP::get_l2_norms_sqr() const {
    return l2_norms_sqr.data();
}

//////////////////////////////////////////////////////////////////////////////////
// IndexHNSWSQ8UniformCosine implementation
//////////////////////////////////////////////////////////////////////////////////

IndexHNSWSQ8UniformCosine::IndexHNSWSQ8UniformCosine() : IndexHNSW() {
    is_cosine = true;
}

IndexHNSWSQ8UniformCosine::IndexHNSWSQ8UniformCosine(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M)
        : IndexHNSW(new IndexScalarQuantizer8bitUniformCosine(d), M) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == ScalarQuantizer::QT_8bit_uniform,
            "IndexHNSWSQ8UniformCosine only supports QT_8bit_uniform");

    is_trained = this->storage->is_trained;
    own_fields = true;
    is_cosine = true;
}

//////////////////////////////////////////////////////////////////////////////////
// IndexHNSWSQ8UniformIP implementation
//////////////////////////////////////////////////////////////////////////////////

IndexHNSWSQ8UniformIP::IndexHNSWSQ8UniformIP() : IndexHNSW() {
    is_cosine = false;
}

IndexHNSWSQ8UniformIP::IndexHNSWSQ8UniformIP(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M)
        : IndexHNSW(new IndexScalarQuantizer8bitUniformIP(d), M) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == ScalarQuantizer::QT_8bit_uniform,
            "IndexHNSWSQ8UniformIP only supports QT_8bit_uniform");

    is_trained = this->storage->is_trained;
    own_fields = true;
    is_cosine = false;
}

} // namespace faiss
