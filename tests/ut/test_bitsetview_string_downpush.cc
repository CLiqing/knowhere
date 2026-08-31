#include <array>
#include <cstdint>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"

namespace {

int32_t
EvaluateSelectedRows(const void* context, const int64_t* row_ids, uint32_t count, uint64_t active_mask,
                     uint64_t* valid_mask) noexcept {
    const auto* selected = static_cast<const std::array<bool, 4>*>(context);
    uint64_t result = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        const auto lane_bit = uint64_t{1} << lane;
        if ((active_mask & lane_bit) == 0) {
            continue;
        }
        const auto row_id = row_ids[lane];
        if (row_id >= 0 && static_cast<size_t>(row_id) < selected->size() && (*selected)[row_id]) {
            result |= lane_bit;
        }
    }
    *valid_mask = result;
    return 0;
}

}  // namespace

TEST_CASE("BitsetView carries an expression-agnostic candidate evaluator") {
    static constexpr std::array<bool, 4> selected = {false, true, false, true};
    static constexpr std::array<uint8_t, 1> mandatory_bits = {0b00000010};

    knowhere::BitsetView::CandidateEvaluatorV1 evaluator;
    evaluator.abi_major = knowhere::BitsetView::kCandidateEvaluatorAbiMajor;
    evaluator.struct_size = sizeof(evaluator);
    evaluator.context = &selected;
    evaluator.eval_batch = &EvaluateSelectedRows;
    evaluator.abi_capabilities = knowhere::BitsetView::kCandidateEvaluatorCapabilityLease;
    evaluator.lease_factory_context = &selected;
    evaluator.acquire_lease = [](const void* context) noexcept -> void* { return const_cast<void*>(context); };
    evaluator.release_lease = [](void*) noexcept {};

    knowhere::BitsetView view(mandatory_bits.data(), selected.size());
    view.set_candidate_evaluator(evaluator, selected.size(), 2);

    CHECK(view.has_candidate_evaluator());
    CHECK(view.base_filtered_out_count() == 1);
    CHECK(view.estimated_count() == 2);
    CHECK(view.test(0));
    CHECK(view.test(1));
    CHECK(view.test(2));
    CHECK_FALSE(view.test(3));

    SECTION("a declared lease must provide a complete acquire/release pair") {
        auto incomplete = evaluator;
        incomplete.release_lease = nullptr;
        knowhere::BitsetView rejected;
        CHECK_THROWS_AS(rejected.set_candidate_evaluator(incomplete, selected.size(), 2), std::invalid_argument);
    }

    SECTION("an original v1 producer remains compatible without lease support") {
        auto original = evaluator;
        original.struct_size = knowhere::BitsetView::kCandidateEvaluatorV1MinimumSize;
        original.abi_capabilities = 0;
        original.lease_factory_context = nullptr;
        original.acquire_lease = nullptr;
        original.release_lease = nullptr;
        knowhere::BitsetView compatible;
        compatible.set_candidate_evaluator(original, selected.size(), 2);
        CHECK_FALSE(compatible.test(1));
    }

    SECTION("a wrapper keeps estimated and exact filtered counts separate") {
        knowhere::BitsetView deferred;
        deferred.set_candidate_evaluator(evaluator, selected.size(), 3);
        REQUIRE(deferred.count() == 0);
        REQUIRE(deferred.base_filtered_out_count() == 0);
        REQUIRE(deferred.estimated_count() == 3);

        knowhere::BitsetView wrapped(deferred.data(), deferred.size(), deferred.count());
        wrapped.copy_candidate_evaluator_from(deferred);
        CHECK(wrapped.count() == 0);
        CHECK(wrapped.base_filtered_out_count() == 0);
        CHECK(wrapped.estimated_count() == 3);
    }
}
