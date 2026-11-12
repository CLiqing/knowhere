// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "knowhere/candidate_evaluator.h"
#include "knowhere/filter_batch_result.h"

namespace knowhere {

// Private execution state owned by a bound filter, not by a search algorithm.
// One independent search task binds once; all its test() calls reuse this state.
// Neither this adapter nor the backend knows expressions or scalar types.
// Not a DSO ABI: only CandidateEvaluatorView crosses the callback boundary.
class CandidateEvaluatorExecution {
 public:
    explicit CandidateEvaluatorExecution(const CandidateEvaluatorView& view) : view_(view) {
        if (!view_.valid()) {
            throw std::invalid_argument("invalid candidate evaluator descriptor");
        }
        if (view_.create_worker(view_.context, &worker_) != CandidateEvalStatus::Success || worker_ == nullptr) {
            if (worker_ != nullptr) {
                view_.destroy_worker(worker_);
            }
            throw std::runtime_error("candidate workspace preparation failed");
        }
    }
    ~CandidateEvaluatorExecution() {
        view_.destroy_worker(worker_);
    }
    CandidateEvaluatorExecution(const CandidateEvaluatorExecution&) = delete;
    CandidateEvaluatorExecution&
    operator=(const CandidateEvaluatorExecution&) = delete;

    int32_t*
    row_buffer(size_t count) {
        mapped_rows_.resize(count);
        return mapped_rows_.data();
    }
    uint64_t*
    active_buffer(size_t count) {
        active_words_.assign(FilterBatchResult::WordCount(count), ~uint64_t{0});
        if (count != 0)
            active_words_.back() &= FilterBatchResult::TailMask(count);
        return active_words_.data();
    }

    static uint64_t
    LaneMask(uint32_t count) {
        if (count > 64) {
            throw std::invalid_argument("candidate batch exceeds 64 lanes");
        }
        return count == 64 ? ~uint64_t{0} : (uint64_t{1} << count) - 1;
    }

    // Public segment offsets, not primary keys. A set return bit means excluded,
    // including inactive lanes. No inactive row is evaluated by the producer.
    uint64_t
    test_word(const int32_t* ids, uint32_t count, uint64_t active) {
        const auto lanes = LaneMask(count);
        if ((active & ~lanes) != 0 || (count != 0 && ids == nullptr)) {
            throw std::invalid_argument("invalid candidate batch IDs/mask");
        }
        if (active == 0) {
            return lanes;
        }
        if (!view_.memoize) {
            uint64_t accepted = 0;
            if (view_.eval_batch(worker_, ids, count, &active, &accepted) != CandidateEvalStatus::Success) {
                throw std::runtime_error("candidate callback failed");
            }
            if ((accepted & ~active) != 0)
                throw std::runtime_error("candidate callback returned inactive lanes");
            ++batch_calls;
            rows += std::popcount(active);
            return lanes & ~accepted;
        }
        // Reserve bucket capacity for the batch. std::unordered_map keeps
        // pointers to stored verdicts valid even if insertion triggers a rehash.
        verdicts_.reserve(verdicts_.size() + std::popcount(active));
        // Values 0/1 are final public-row verdicts; 2..65 name the first lane
        // of a same-batch miss. Group OR and mandatory visibility stay outside.
        uint64_t accepted = 0, missing = 0, duplicates = 0;
        std::array<uint8_t*, 64> slots;
        std::array<uint8_t, 64> source_lane;
        try {
            for (auto remaining = active; remaining != 0; remaining &= remaining - 1) {
                const auto lane = std::countr_zero(remaining);
                const auto [entry, inserted] = verdicts_.try_emplace(ids[lane], static_cast<uint8_t>(lane + 2));
                if (inserted) {
                    missing |= uint64_t{1} << lane;
                    slots[lane] = &entry->second;
                } else if (entry->second < 2) {
                    accepted |= uint64_t{entry->second} << lane;
                } else {
                    duplicates |= uint64_t{1} << lane;
                    source_lane[lane] = entry->second - 2;
                }
            }
            if (missing != 0) {
                uint64_t produced = 0;
                if (view_.eval_batch(worker_, ids, count, &missing, &produced) != CandidateEvalStatus::Success) {
                    throw std::runtime_error("candidate callback failed");
                }
                if ((produced & ~missing) != 0) {
                    throw std::runtime_error("candidate callback returned inactive lanes");
                }
                for (auto remaining = missing; remaining != 0; remaining &= remaining - 1) {
                    const auto lane = std::countr_zero(remaining);
                    *slots[lane] = (produced >> lane) & 1;
                }
                accepted |= produced;
                // Only duplicates need positional restoration. First misses
                // already occupy their original lanes in the producer result.
                for (auto remaining = duplicates; remaining != 0; remaining &= remaining - 1) {
                    const auto lane = std::countr_zero(remaining);
                    accepted |= ((produced >> source_lane[lane]) & 1) << lane;
                }
                ++batch_calls;
                rows += std::popcount(missing);
            }
        } catch (...) {
            // Errors are not FALSE. Remove this batch's provisional entries so
            // a caller that handles an error cannot observe unresolved verdicts.
            for (auto remaining = missing; remaining != 0; remaining &= remaining - 1) {
                verdicts_.erase(ids[std::countr_zero(remaining)]);
            }
            throw;
        }
        return lanes & ~accepted;
    }

    // Arbitrary-length transport. The common <=64 memoized kernel above stays
    // free of result-buffer allocation; a post batch bypasses the verdict map.
    FilterBatchResult
    test(const int32_t* ids, size_t count, const uint64_t* active) {
        if (count != 0 && ids == nullptr)
            throw std::invalid_argument("invalid candidate batch IDs");
        if (count <= 64) {
            return FilterBatchResult::FromWord(
                count, test_word(ids, static_cast<uint32_t>(count),
                                 active == nullptr ? LaneMask(static_cast<uint32_t>(count)) : active[0]));
        }
        FilterBatchResult excluded(count, true);
        if (!view_.memoize) {
            // The producer writes accepted bits directly into the owned return
            // buffer; validate/invert in place instead of allocating another.
            std::fill_n(excluded.data(), FilterBatchResult::WordCount(count), uint64_t{0});
            if (view_.eval_batch(worker_, ids, count, active, excluded.data()) != CandidateEvalStatus::Success)
                throw std::runtime_error("candidate callback failed");
            size_t evaluated = 0;
            for (size_t word = 0; word < FilterBatchResult::WordCount(count); ++word) {
                const auto mask =
                    word + 1 == FilterBatchResult::WordCount(count) ? FilterBatchResult::TailMask(count) : ~uint64_t{0};
                const auto enabled = active == nullptr ? mask : active[word] & mask;
                if ((excluded.data()[word] & ~enabled) != 0)
                    throw std::runtime_error("candidate callback returned inactive lanes");
                excluded.data()[word] = mask & ~excluded.data()[word];
                evaluated += std::popcount(enabled);
            }
            ++batch_calls;
            rows += evaluated;
            return excluded;
        }
        // Only the large memoized case needs a batch-local index. Do not put
        // lane positions into the byte-sized final-verdict cache.
        std::vector<int32_t> missing;
        std::unordered_map<int32_t, size_t> pending;
        for (size_t lane = 0; lane < count; ++lane) {
            if (active != nullptr && ((active[lane / 64] >> (lane % 64)) & 1) == 0)
                continue;
            if (verdicts_.find(ids[lane]) == verdicts_.end() && pending.try_emplace(ids[lane], missing.size()).second)
                missing.push_back(ids[lane]);
        }
        if (!missing.empty()) {
            FilterBatchResult accepted(missing.size());
            if (view_.eval_batch(worker_, missing.data(), missing.size(), nullptr, accepted.data()) !=
                CandidateEvalStatus::Success)
                throw std::runtime_error("candidate callback failed");
            if ((accepted.data()[FilterBatchResult::WordCount(missing.size()) - 1] &
                 ~FilterBatchResult::TailMask(missing.size())) != 0)
                throw std::runtime_error("candidate callback returned out-of-domain lanes");
            for (size_t i = 0; i < missing.size(); ++i) verdicts_.insert_or_assign(missing[i], uint8_t{accepted[i]});
            ++batch_calls;
            rows += missing.size();
        }
        for (size_t lane = 0; lane < count; ++lane) {
            if (active == nullptr || ((active[lane / 64] >> (lane % 64)) & 1) != 0)
                excluded.set(lane, verdicts_.at(ids[lane]) == 0);
        }
        return excluded;
    }

    size_t batch_calls = 0;
    // Successfully evaluated distinct public rows in this independent task.
    size_t rows = 0;

 private:
    CandidateEvaluatorView view_;
    void* worker_ = nullptr;
    std::unordered_map<int32_t, uint8_t> verdicts_;
    std::vector<int32_t> mapped_rows_;
    std::vector<uint64_t> active_words_;
};
}  // namespace knowhere
