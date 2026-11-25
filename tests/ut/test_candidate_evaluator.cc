// Licensed under the Apache License, Version 2.0.
#include "catch2/catch_test_macros.hpp"
#include "knowhere/candidate_evaluator.h"

TEST_CASE("Independent evaluator owns one worker and accepts multiword batches", "[candidate]") {
    using namespace knowhere;
    unsigned destroys = 0;
    CandidateEvaluatorView view;
    view.row_count = 65;
    view.context = &destroys;
    view.create_worker = [](const void* ctx, void** out) noexcept {
        *out = const_cast<void*>(ctx);
        return CandidateEvalStatus::Success;
    };
    view.destroy_worker = [](void* worker) noexcept { ++*static_cast<unsigned*>(worker); };
    view.eval_batch = [](void*, const int32_t*, size_t n, const uint64_t* active, uint64_t* out) noexcept {
        if (n != 65 || !active)
            return CandidateEvalStatus::InvalidArgument;
        out[0] = active[0];
        out[1] = active[1];
        return CandidateEvalStatus::Success;
    };
    {
        CandidateEvaluatorExecution task(view);
        REQUIRE(task.Bind() == CandidateEvalStatus::Success);
        REQUIRE(task.Bind() == CandidateEvalStatus::InvalidArgument);
        int32_t ids[65]{};
        uint64_t mask[]{0, 1}, accepted[2]{};
        REQUIRE(task.Evaluate(ids, 65, mask, accepted) == CandidateEvalStatus::Success);
        REQUIRE(accepted[0] == 0);
        REQUIRE(accepted[1] == 1);
    }
    REQUIRE(destroys == 1);
    view.eval_batch = nullptr;
    REQUIRE_FALSE(view.valid());
}
