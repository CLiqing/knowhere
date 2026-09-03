// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace knowhere {

// CandidateEvaluatorV1 is a same-process, cross-DSO C ABI. Increment the
// major only when an existing field, callback signature, or semantic contract
// changes incompatibly. Append-only optional fields keep major version 1 and
// are discovered through struct_size plus abi_capabilities.
inline constexpr uint32_t kCandidateEvaluatorAbiMajor = 1;
inline constexpr uint64_t kCandidateEvaluatorCapabilityLease = 1ULL << 0;
inline constexpr uint64_t kCandidateEvaluatorCapabilityWorkerWorkspace = 1ULL << 1;

// Evaluates arbitrary logical segment row IDs.
//
// context: immutable Milvus-owned evaluator state.
// row_ids: `count` logical segment row IDs; the callback does not receive
//          Cardinal internal graph IDs.
// count: number of lanes in [0, 64].
// active_mask: input lanes to evaluate; bits >= count must be zero.
// accepted_mask: output lanes accepted by the complete predicate; it must be
//                a subset of active_mask and bits >= count must be zero.
// return: zero on success, non-zero on failure. No exception may cross this
//         noexcept boundary.
using CandidateEvalBatchFn = int32_t (*)(const void* context, const int64_t* row_ids, uint32_t count,
                                         uint64_t active_mask, uint64_t* accepted_mask) noexcept;

// Evaluates the contiguous logical row-ID range
// [first_row_id, first_row_id + count) under the same mask/result contract as
// CandidateEvalBatchFn. This optional fast path is used by scan completion.
using CandidateEvalContiguousFn = int32_t (*)(const void* context, int64_t first_row_id, uint32_t count,
                                              uint64_t active_mask, uint64_t* accepted_mask) noexcept;

// Maps an index-internal/physical row ID to the logical segment row-ID domain
// expected by the evaluator; a negative result rejects the candidate.
using CandidateEvaluatorRowIdMapperFn = int64_t (*)(const void* context, int64_t internal_row_id) noexcept;

// Optional iterator ownership protocol. acquire_lease returns an opaque owned
// lease retained until the matching release_lease call.
using CandidateEvaluatorAcquireLeaseFn = void* (*)(const void* context) noexcept;
using CandidateEvaluatorReleaseLeaseFn = void (*)(void* lease) noexcept;

// Optional per-worker mutable scratch. The prepared evaluator context remains
// immutable and query-scoped; every concurrent search worker and every
// iterator workspace creates and owns one independent workspace.
using CandidateEvaluatorCreateWorkspaceFn = void* (*)(const void* context) noexcept;
using CandidateEvaluatorReleaseWorkspaceFn = void (*)(void* workspace) noexcept;
using CandidateEvalBatchWithWorkspaceFn = int32_t (*)(const void* context, void* workspace, const int64_t* row_ids,
                                                      uint32_t count, uint64_t active_mask,
                                                      uint64_t* accepted_mask) noexcept;
using CandidateEvalContiguousWithWorkspaceFn = int32_t (*)(const void* context, void* workspace, int64_t first_row_id,
                                                           uint32_t count, uint64_t active_mask,
                                                           uint64_t* accepted_mask) noexcept;

struct CandidateEvaluatorV1 {
    uint32_t abi_major = kCandidateEvaluatorAbiMajor;
    uint32_t struct_size = sizeof(CandidateEvaluatorV1);
    uint64_t abi_capabilities = 0;
    const void* context = nullptr;
    CandidateEvalBatchFn eval_batch = nullptr;
    CandidateEvalContiguousFn eval_contiguous = nullptr;
    const void* row_id_mapper_context = nullptr;
    CandidateEvaluatorRowIdMapperFn row_id_mapper = nullptr;
    const void* lease_factory_context = nullptr;
    CandidateEvaluatorAcquireLeaseFn acquire_lease = nullptr;
    CandidateEvaluatorReleaseLeaseFn release_lease = nullptr;
    CandidateEvaluatorCreateWorkspaceFn create_workspace = nullptr;
    CandidateEvaluatorReleaseWorkspaceFn release_workspace = nullptr;
    CandidateEvalBatchWithWorkspaceFn eval_batch_with_workspace = nullptr;
    CandidateEvalContiguousWithWorkspaceFn eval_contiguous_with_workspace = nullptr;
};

static_assert(std::is_trivially_copyable_v<CandidateEvaluatorV1>);
static_assert(std::is_standard_layout_v<CandidateEvaluatorV1>);

// Fields before lease_factory_context form the required V1 prefix. Lease
// fields were appended as an optional capability and must not be read unless
// both struct_size and the capability bit declare them.
inline constexpr size_t kCandidateEvaluatorV1MinimumSize = offsetof(CandidateEvaluatorV1, lease_factory_context);
static_assert(kCandidateEvaluatorV1MinimumSize <= sizeof(CandidateEvaluatorV1));

}  // namespace knowhere
