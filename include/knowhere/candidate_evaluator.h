// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace knowhere {

enum class CandidateEvalStatus : uint32_t { Success, InvalidArgument, Failed };

// Borrowed during a synchronous search; worker state is private to a search task.
// Offsets address public rows. Masks use ceil(count/64) words, low bit first;
// null active means all rows. The producer writes only active, predicate-true lanes.
struct CandidateEvaluatorView {
    uint32_t abi_major = 2;
    uint32_t struct_size = sizeof(CandidateEvaluatorView);
    size_t row_count = 0;
    const void* context = nullptr;
    CandidateEvalStatus (*create_worker)(const void*, void**) noexcept = nullptr;
    void (*destroy_worker)(void*) noexcept = nullptr;
    CandidateEvalStatus (*eval_batch)(void*, const int32_t*, size_t, const uint64_t*, uint64_t*) noexcept = nullptr;

    bool
    valid() const noexcept {
        return abi_major == 2 && struct_size == sizeof(CandidateEvaluatorView) && context && create_worker &&
               destroy_worker && eval_batch;
    }
};
static_assert(std::is_standard_layout_v<CandidateEvaluatorView> &&
              std::is_trivially_copyable_v<CandidateEvaluatorView>);

class CandidateEvaluatorExecution {
 public:
    explicit CandidateEvaluatorExecution(const CandidateEvaluatorView& view) noexcept : view_(view) {
    }
    CandidateEvaluatorExecution(const CandidateEvaluatorExecution&) = delete;
    CandidateEvaluatorExecution&
    operator=(const CandidateEvaluatorExecution&) = delete;

    CandidateEvalStatus
    Bind() noexcept {
        if (!view_.valid() || worker_)
            return CandidateEvalStatus::InvalidArgument;
        void* worker = nullptr;
        auto status = view_.create_worker(view_.context, &worker);
        if (status != CandidateEvalStatus::Success || !worker) {
            if (worker)
                view_.destroy_worker(worker);
            return status == CandidateEvalStatus::Success ? CandidateEvalStatus::Failed : status;
        }
        worker_ = worker;
        return status;
    }

    CandidateEvalStatus
    Evaluate(const int32_t* ids, size_t count, const uint64_t* active, uint64_t* accepted) const noexcept {
        if (!worker_)
            return CandidateEvalStatus::InvalidArgument;
        if (!count)
            return CandidateEvalStatus::Success;
        if (!ids || !accepted)
            return CandidateEvalStatus::InvalidArgument;
        return view_.eval_batch(worker_, ids, count, active, accepted);
    }

    ~CandidateEvaluatorExecution() {
        if (worker_)
            view_.destroy_worker(worker_);
    }

 private:
    CandidateEvaluatorView view_;
    void* worker_ = nullptr;
};
}  // namespace knowhere
