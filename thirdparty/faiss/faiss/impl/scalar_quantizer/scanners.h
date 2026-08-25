/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Private implementation header for per-SIMD scalar quantizer TUs.
// Do not include in public APIs.

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/simd_levels.h>

#include <faiss/impl/simd_dispatch.h>

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

template <typename C, class DCClass>
size_t run_scan_codes_batch_4(
        const DCClass& dc,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        size_t code_size,
        size_t list_no,
        bool store_pairs,
        float distance_offset,
        ResultHandler& handler) {
    size_t nup = 0;
    float threshold = handler.threshold;
    size_t j = 0;
    for (; j + 4 <= list_size; j += 4) {
        float distances[4];
        dc.query_to_codes_batch_4(
                codes,
                codes + code_size,
                codes + 2 * code_size,
                codes + 3 * code_size,
                distances[0],
                distances[1],
                distances[2],
                distances[3]);
        handler.stats.scan_cnt += 4;
        for (size_t lane = 0; lane < 4; ++lane) {
            const float distance = distance_offset + distances[lane];
            if (C::cmp(threshold, distance)) {
                const int64_t id = store_pairs ? lo_build(list_no, j + lane)
                                               : ids[j + lane];
                if (handler.add_result(distance, id)) {
                    handler.stats.nheap_updates++;
                    ++nup;
                    threshold = handler.threshold;
                }
            }
        }
        codes += 4 * code_size;
    }
    for (; j < list_size; ++j) {
        handler.stats.scan_cnt++;
        const float distance = distance_offset + dc.query_to_code(codes);
        if (C::cmp(threshold, distance)) {
            const int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            if (handler.add_result(distance, id)) {
                handler.stats.nheap_updates++;
                ++nup;
                threshold = handler.threshold;
            }
        }
        codes += code_size;
    }
    return nup;
}

/*******************************************************************
 * IVFSQScannerIP / IVFSQScannerL2 — moved from anonymous namespace
 * in ScalarQuantizer.cpp
 *******************************************************************/

template <class DCClass>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;
    bool use_batch_4;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual,
            bool use_batch_4)
            : dc(d, trained),
              by_residual(by_residual),
              use_batch_4(use_batch_4),
              accu0(0) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
        this->keep_max = true;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        if (use_batch_4 && this->sel == nullptr) {
            return run_scan_codes_batch_4<CMin<float, idx_t>>(
                    dc,
                    list_size,
                    codes,
                    ids,
                    this->code_size,
                    this->list_no,
                    this->store_pairs,
                    accu0,
                    handler);
        }
        return run_scan_codes_fix_C<CMin<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

template <class DCClass>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    bool use_batch_4;
    const Index* quantizer;
    const float* x; /// current query

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual,
            bool use_batch_4)
            : dc(d, trained),
              by_residual(by_residual),
              use_batch_4(use_batch_4),
              quantizer(quantizer),
              x(nullptr),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (by_residual) {
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        if (use_batch_4 && this->sel == nullptr) {
            return run_scan_codes_batch_4<CMax<float, idx_t>>(
                    dc,
                    list_size,
                    codes,
                    ids,
                    this->code_size,
                    this->list_no,
                    this->store_pairs,
                    0.0f,
                    handler);
        }
        return run_scan_codes_fix_C<CMax<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

/*******************************************************************
 * Forward declaration of inverts list scanner
 *******************************************************************/

template <SIMDLevel SL>
InvertedListScanner* sq_select_InvertedListScanner(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

/// Scanner for QT_0bit / centroid-only distance: always returns the
/// coarse distance that was set via set_list().
struct IVFCoarseDistanceScanner : InvertedListScanner {
    float coarse_dis = 0;

    IVFCoarseDistanceScanner(
            bool is_similarity,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel) {
        code_size = 0;
        keep_max = is_similarity;
    }

    void set_query(const float* /*query_vector*/) override {}

    void set_list(idx_t list_no_in, float coarse_dis_in) override {
        this->list_no = list_no_in;
        this->coarse_dis = coarse_dis_in;
    }

    float distance_to_code(const uint8_t* /*code*/) const override {
        return coarse_dis;
    }
};

} // namespace scalar_quantizer

} // namespace faiss
