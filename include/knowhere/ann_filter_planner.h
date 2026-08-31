// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace knowhere {

inline constexpr uint32_t kAnnFilterPlannerAbiMajor = 1;

enum class AnnFilterRequestMode : uint32_t {
    kExplicitFusing = 0,
    kAuto = 1,
};

enum class AnnFilterPolicy : uint32_t {
    kBaseline = 0,
    kFusing = 1,
};

enum class AnnFilterPlanReason : uint32_t {
    kNone = 0,
    kPlannerUnavailable = 1,
    kGraphUnavailable = 2,
    kCostBaseline = 3,
    kIncompatibleRequest = 4,
};

// Public, expression-agnostic inputs. Predicate execution remains in the
// query-scoped CandidateEvaluator; Cardinal sees only stable cost features.
struct AnnFilterPlanRequestV1 {
    uint32_t abi_major = kAnnFilterPlannerAbiMajor;
    uint32_t struct_size = sizeof(AnnFilterPlanRequestV1);
    AnnFilterRequestMode mode = AnnFilterRequestMode::kExplicitFusing;
    uint32_t reserved = 0;
    uint64_t row_count = 0;
    uint64_t estimated_filtered_out_count = 0;
    uint64_t nq = 0;
    uint64_t topk = 0;
    uint32_t predicate_leaf_count = 0;
    uint32_t predicate_logical_node_count = 0;
};

struct AnnFilterPlanResultV1 {
    uint32_t abi_major = kAnnFilterPlannerAbiMajor;
    uint32_t struct_size = sizeof(AnnFilterPlanResultV1);
    AnnFilterPolicy policy = AnnFilterPolicy::kBaseline;
    AnnFilterPlanReason reason = AnnFilterPlanReason::kPlannerUnavailable;
};

static_assert(std::is_standard_layout_v<AnnFilterPlanRequestV1>);
static_assert(std::is_trivially_copyable_v<AnnFilterPlanRequestV1>);
static_assert(std::is_standard_layout_v<AnnFilterPlanResultV1>);
static_assert(std::is_trivially_copyable_v<AnnFilterPlanResultV1>);

// Optional capability implemented only by backends that own an ANN fusing
// plan. Keeping this separate avoids changing IndexNode's cross-DSO vtable.
class AnnFilterPlanner {
 public:
    virtual ~AnnFilterPlanner() = default;

    virtual AnnFilterPlanResultV1
    PlanAnnFilter(const AnnFilterPlanRequestV1& request) const noexcept = 0;
};

}  // namespace knowhere
