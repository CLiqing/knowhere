// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef BITSET_H
#define BITSET_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace knowhere {
class BitsetView {
 public:
    enum class ExtraScalarInt64PredicateOp : int32_t {
        kNone = 0,
        kGreaterEqual = 1,
        kModLessThan = 2,
        kGreaterThan = 3,
        kLessEqual = 4,
        kLessThan = 5,
        kEqual = 6,
        kNotEqual = 7,
    };

    struct ExtraScalarInt64PredicateFilter {
        const int64_t* row_values = nullptr;
        const int64_t* const* chunk_values = nullptr;
        const int64_t* chunk_offsets = nullptr;
        size_t num_chunks = 0;
        size_t row_count = 0;
        ExtraScalarInt64PredicateOp op = ExtraScalarInt64PredicateOp::kNone;
        int64_t arg0 = 0;
        int64_t arg1 = 0;
    };

    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits, size_t num_filtered_out_bits = 0, size_t id_offset = 0)
        : bits_(data), num_bits_(num_bits), num_filtered_out_bits_(num_filtered_out_bits), id_offset_(id_offset) {
    }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    bool
    empty() const {
        return num_bits_ == 0;
    }

    // return the number of the bits. if with id mapping, return the number of the internal ids.
    size_t
    size() const {
        if (out_ids_ != nullptr) {
            return num_internal_ids_;
        }
        return num_bits_;
    }

    // return the number of filtered out bits. if with id mapping, return the number of filtered out ids.
    size_t
    count() const {
        if (out_ids_ != nullptr) {
            return num_filtered_out_ids_;
        }
        return num_filtered_out_bits_;
    }

    size_t
    estimated_count() const {
        if (!has_extra_scalar_int64_predicate_filter_) {
            return count();
        }
        return std::min(size(), std::max(count(), extra_filtered_out_count_));
    }

    size_t
    byte_size() const {
        return (num_bits_ + 8 - 1) >> 3;
    }

    const uint8_t*
    data() const {
        return bits_;
    }

    bool
    has_out_ids() const {
        return out_ids_ != nullptr;
    }

    void
    set_out_ids(const uint32_t* out_ids, size_t num_internal_ids,
                std::optional<size_t> num_filtered_out_ids = std::nullopt) {
        out_ids_ = out_ids;
        num_internal_ids_ = num_internal_ids;
        if (num_filtered_out_ids.has_value()) {
            num_filtered_out_ids_ = num_filtered_out_ids.value();
        } else {
            // auto calculate num_filtered_out_ids if not provided
            num_filtered_out_ids_ = get_filtered_out_num_();
        }
    }

    const uint32_t*
    out_ids_data() const {
        if (out_ids_ == nullptr) {
            return nullptr;
        }
        return out_ids_;
    }

    void
    set_id_offset(size_t id_offset) {
        id_offset_ = id_offset;
    }

    void
    set_extra_scalar_int64_predicate_filter(const ExtraScalarInt64PredicateFilter& filter,
                                            size_t estimated_filtered_out_count) {
        extra_scalar_int64_predicate_filter_ = filter;
        has_extra_scalar_int64_predicate_filter_ = true;
        extra_filtered_out_count_ = estimated_filtered_out_count;
    }

    bool
    has_extra_scalar_int64_predicate_filter() const {
        return has_extra_scalar_int64_predicate_filter_;
    }

    const ExtraScalarInt64PredicateFilter&
    extra_scalar_int64_predicate_filter() const {
        return extra_scalar_int64_predicate_filter_;
    }

    size_t
    extra_filtered_out_count() const {
        return extra_filtered_out_count_;
    }

    // if the test succeeds, then the index should be skipped during search; otherwise, it should be included.
    bool
    test(int64_t index) const {
        int64_t out_id = index + id_offset_;
        if (out_ids_ != nullptr) {
            out_id = out_ids_[out_id];
        }
        // when index is larger than the max_offset, ignore it
        bool filtered = (out_id >= static_cast<int64_t>(num_bits_)) || (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
        if (!filtered && has_extra_scalar_int64_predicate_filter_) {
            filtered = test_extra_scalar_int64_predicate_filter_(out_id);
        }
        return filtered;
    }
    // return the filtered ratio. if with id mapping, calculated by internal_ids rather than bits.
    float
    filter_ratio() const {
        return empty() ? 0.0f : ((float)estimated_count() / size());
    }

    size_t
    get_filtered_out_num_() const {
        if (empty()) {
            return 0;
        }
        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            size_t count = 0;
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (test(i)) {
                    count++;
                }
            }
            return count;
        }
        // if without id mapping, use a better algorithm to calculate the number of filtered out bits.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        auto popcount8 = [&](uint8_t x) -> int {
            x = (x & 0x55) + ((x >> 1) & 0x55);
            x = (x & 0x33) + ((x >> 2) & 0x33);
            x = (x & 0x0F) + ((x >> 4) & 0x0F);
            return x;
        };

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            ret += __builtin_popcountll(*p_uint64);
            p_uint64++;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = (len_uint64 << 3); i < len_uint8; i++) {
            ret += popcount8(*p_uint8);
            p_uint8++;
        }

        return ret;
    }

    // return the first valid idx. if with id mapping, return the first valid internal_id.
    size_t
    get_first_valid_index() const {
        if (has_extra_scalar_int64_predicate_filter_) {
            for (size_t i = 0; i < size(); i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return size();
        }

        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return num_internal_ids_;
        }
        // if without id mapping, use a better algorithm to find the first valid index.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            uint64_t value = (~(*p_uint64));
            if (value == 0) {
                p_uint64++;
                continue;
            }
            ret = __builtin_ctzll(value);
            return i * 64 + ret;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = 0; i < len_uint8 - (len_uint64 << 3); i++) {
            uint8_t value = (~(*p_uint8));
            if (value == 0) {
                p_uint8++;
                continue;
            }
            ret = __builtin_ctz(value);
            return len_uint64 * 64 + i * 8 + ret;
        }

        return num_bits_;
    }

    std::string
    to_string(size_t from, size_t to) const {
        if (empty()) {
            return "";
        }
        std::stringbuf buf;
        to = std::min<size_t>(to, num_bits_);
        for (size_t i = from; i < to; i++) {
            buf.sputc(test(i) ? '1' : '0');
        }
        return buf.str();
    }

 private:
    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
    size_t num_filtered_out_bits_ = 0;

    // optional. many indexes will share one bitset, requiring offset to distinguish between them.
    //  like multi-chunk brute-force in /src/common/comp/brute_force.cc, or mv-only in /src/index/hnsw/faiss_hnsw.cc
    size_t id_offset_ = 0;  // offset of the internal ids

    // optional. bitset supports id mapping.
    // Even allows multiple ids to map to the same bit, so the number of internal ids and bits may be not equal.
    const uint32_t* out_ids_ = nullptr;
    size_t num_internal_ids_ = 0;
    size_t num_filtered_out_ids_ = 0;

    size_t extra_filtered_out_count_ = 0;
    ExtraScalarInt64PredicateFilter extra_scalar_int64_predicate_filter_;
    bool has_extra_scalar_int64_predicate_filter_ = false;

    bool
    test_extra_scalar_int64_predicate_filter_(int64_t out_id) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
        int64_t value = 0;
        if (!get_extra_scalar_int64_predicate_value_(out_id, &value)) {
            return true;
        }
        switch (filter.op) {
            case ExtraScalarInt64PredicateOp::kGreaterEqual:
                return value < filter.arg0;
            case ExtraScalarInt64PredicateOp::kGreaterThan:
                return value <= filter.arg0;
            case ExtraScalarInt64PredicateOp::kLessEqual:
                return value > filter.arg0;
            case ExtraScalarInt64PredicateOp::kLessThan:
                return value >= filter.arg0;
            case ExtraScalarInt64PredicateOp::kEqual:
                return value != filter.arg0;
            case ExtraScalarInt64PredicateOp::kNotEqual:
                return value == filter.arg0;
            case ExtraScalarInt64PredicateOp::kModLessThan:
                return filter.arg0 <= 0 || value % filter.arg0 >= filter.arg1;
            case ExtraScalarInt64PredicateOp::kNone:
                break;
        }
        return true;
    }

    bool
    get_extra_scalar_int64_predicate_value_(int64_t out_id, int64_t* value) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
        if (value == nullptr || out_id < 0 || static_cast<size_t>(out_id) >= filter.row_count) {
            return false;
        }
        if (filter.row_values != nullptr) {
            *value = filter.row_values[out_id];
            return true;
        }
        if (filter.chunk_values != nullptr && filter.chunk_offsets != nullptr && filter.num_chunks > 0) {
            if (filter.num_chunks == 1) {
                *value = filter.chunk_values[0][out_id];
                return true;
            }
            const auto* upper =
                std::upper_bound(filter.chunk_offsets, filter.chunk_offsets + filter.num_chunks + 1, out_id);
            if (upper == filter.chunk_offsets) {
                return false;
            }
            const auto chunk_idx = static_cast<size_t>((upper - filter.chunk_offsets) - 1);
            if (chunk_idx >= filter.num_chunks) {
                return false;
            }
            *value = filter.chunk_values[chunk_idx][out_id - filter.chunk_offsets[chunk_idx]];
            return true;
        }
        return false;
    }
};
}  // namespace knowhere

#endif /* BITSET_H */
