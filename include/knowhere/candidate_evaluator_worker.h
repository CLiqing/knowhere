// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstddef>
#include <stdexcept>
#include "knowhere/candidate_evaluator.h"

namespace knowhere {

// One search worker owns one opaque, mutable producer workspace. Neither the
// backend nor this adapter knows expressions or scalar types. Not a DSO ABI.
class CandidateEvaluatorWorker {
 public:
    explicit CandidateEvaluatorWorker(const CandidateEvaluatorViewV1& view) : view_(view) {
        if (!view_.valid() || view_.create_worker(view_.context, &worker_) != CandidateEvalStatus::Success ||
            worker_ == nullptr) {
            if (worker_ != nullptr && view_.valid()) {
                view_.destroy_worker(worker_);
            }
            throw std::runtime_error("ann_fusing: candidate workspace preparation failed");
        }
    }
    ~CandidateEvaluatorWorker() { view_.destroy_worker(worker_); }
    CandidateEvaluatorWorker(const CandidateEvaluatorWorker&) = delete;
    CandidateEvaluatorWorker& operator=(const CandidateEvaluatorWorker&) = delete;

    static uint64_t LaneMask(uint32_t count) {
        if (count > 64) {
            throw std::invalid_argument("ann_fusing: batch exceeds 64 lanes");
        }
        return count == 64 ? ~uint64_t{0} : (uint64_t{1} << count) - 1;
    }

    // Public segment offsets, not primary keys. A set return bit means excluded,
    // including inactive lanes. No inactive row is evaluated by the producer.
    uint64_t test(const int32_t* ids, uint32_t count, uint64_t active) {
        const auto lanes = LaneMask(count);
        if ((active & ~lanes) != 0 || (count != 0 && ids == nullptr)) {
            throw std::invalid_argument("ann_fusing: invalid batch IDs/mask");
        }
        if (active == 0) {
            return lanes;
        }
        uint64_t accepted = 0;
        if (view_.eval_batch(worker_, ids, count, active, &accepted) != CandidateEvalStatus::Success) {
            throw std::runtime_error("ann_fusing: candidate callback failed");
        }
        if ((accepted & ~active) != 0) {
            throw std::runtime_error("ann_fusing: callback returned inactive lanes");
        }
        ++batch_calls;
        rows += count;
        return lanes & ~accepted;
    }

    size_t batch_calls = 0;
    size_t rows = 0;

 private:
    CandidateEvaluatorViewV1 view_;
    void* worker_ = nullptr;
};
}  // namespace knowhere
