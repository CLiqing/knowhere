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
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

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
        kRange = 8,
        kAddLessThan = 9,
        kTerm = 10,
        kSubLessThan = 11,
        kMulLessThan = 12,
        kDivLessThan = 13,
        kPrefixMatch = 14,
        kPostfixMatch = 15,
        kInnerMatch = 16,
        kLikeMatch = 17,
    };

    enum class ExtraScalarPredicateValueType : int32_t {
        kInt64 = 0,
        kFloat = 1,
        kString = 2,
    };

    struct RawStringColumnView {
        const char* const* chunk_bases = nullptr;
        const uint32_t* const* chunk_value_offsets = nullptr;
        const bool* const* chunk_valid_data = nullptr;
        const size_t* chunk_row_counts = nullptr;
        const int64_t* chunk_row_offsets = nullptr;
        size_t num_chunks = 0;
        size_t row_count = 0;
        size_t uniform_chunk_rows = 0;
    };

    struct CompiledLikePatternView {
        const uint32_t* token_offsets = nullptr;
        const uint32_t* token_sizes = nullptr;
        const uint8_t* token_types = nullptr;
        size_t token_count = 0;
    };

    static_assert(std::is_trivially_copyable_v<RawStringColumnView>);
    static_assert(std::is_standard_layout_v<RawStringColumnView>);
    static_assert(std::is_trivially_copyable_v<CompiledLikePatternView>);
    static_assert(std::is_standard_layout_v<CompiledLikePatternView>);

    struct ExtraScalarInt64PredicateFilter {
        ExtraScalarPredicateValueType value_type = ExtraScalarPredicateValueType::kInt64;
        const int64_t* row_values = nullptr;
        const int64_t* const* chunk_values = nullptr;
        const int64_t* chunk_offsets = nullptr;
        size_t num_chunks = 0;
        size_t row_count = 0;
        const float* row_float_values = nullptr;
        const float* const* chunk_float_values = nullptr;
        RawStringColumnView string_column;
        ExtraScalarInt64PredicateOp op = ExtraScalarInt64PredicateOp::kNone;
        int64_t arg0 = 0;
        int64_t arg1 = 0;
        double double_arg0 = 0.0;
        double double_arg1 = 0.0;
        const char* string_arg0_data = nullptr;
        uint32_t string_arg0_size = 0;
        const char* string_arg1_data = nullptr;
        uint32_t string_arg1_size = 0;
        CompiledLikePatternView like_pattern;
        const int64_t* int64_terms = nullptr;
        size_t int64_term_count = 0;
        const double* double_terms = nullptr;
        size_t double_term_count = 0;
        const char* const* string_term_values = nullptr;
        const uint32_t* string_term_sizes = nullptr;
        size_t string_term_count = 0;
        bool string_terms_sorted = false;
        bool lower_inclusive = true;
        bool upper_inclusive = true;
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

    void
    copy_extra_scalar_int64_predicate_filter_from(const BitsetView& other) {
        if (other.has_extra_scalar_int64_predicate_filter_) {
            set_extra_scalar_int64_predicate_filter(other.extra_scalar_int64_predicate_filter_,
                                                    other.extra_filtered_out_count_);
        }
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
        switch (filter.value_type) {
            case ExtraScalarPredicateValueType::kInt64: {
                int64_t value = 0;
                if (!get_extra_scalar_int64_predicate_value_(out_id, &value)) {
                    return true;
                }
                return test_int64_predicate_(value);
            }
            case ExtraScalarPredicateValueType::kFloat: {
                float value = 0.0F;
                if (!get_extra_scalar_float_predicate_value_(out_id, &value)) {
                    return true;
                }
                return test_double_predicate_(static_cast<double>(value));
            }
            case ExtraScalarPredicateValueType::kString: {
                std::string_view value;
                bool is_valid = false;
                if (!get_extra_scalar_string_predicate_value_(out_id, &value, &is_valid) || !is_valid) {
                    return true;
                }
                return test_string_predicate_(value);
            }
        }
        return true;
    }

    bool
    test_int64_predicate_(int64_t value) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
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
            case ExtraScalarInt64PredicateOp::kAddLessThan:
                return static_cast<__int128>(value) + static_cast<__int128>(filter.arg0) >=
                       static_cast<__int128>(filter.arg1);
            case ExtraScalarInt64PredicateOp::kSubLessThan:
                return static_cast<__int128>(value) - static_cast<__int128>(filter.arg0) >=
                       static_cast<__int128>(filter.arg1);
            case ExtraScalarInt64PredicateOp::kMulLessThan:
                return static_cast<__int128>(value) * static_cast<__int128>(filter.arg0) >=
                       static_cast<__int128>(filter.arg1);
            case ExtraScalarInt64PredicateOp::kDivLessThan:
                return filter.arg0 == 0 || value / filter.arg0 >= filter.arg1;
            case ExtraScalarInt64PredicateOp::kTerm:
                for (size_t i = 0; i < filter.int64_term_count; ++i) {
                    if (value == filter.int64_terms[i]) {
                        return false;
                    }
                }
                return true;
            case ExtraScalarInt64PredicateOp::kRange: {
                const bool lower_ok = filter.lower_inclusive ? value >= filter.arg0 : value > filter.arg0;
                const bool upper_ok = filter.upper_inclusive ? value <= filter.arg1 : value < filter.arg1;
                return !(lower_ok && upper_ok);
            }
            case ExtraScalarInt64PredicateOp::kPrefixMatch:
            case ExtraScalarInt64PredicateOp::kPostfixMatch:
            case ExtraScalarInt64PredicateOp::kInnerMatch:
            case ExtraScalarInt64PredicateOp::kLikeMatch:
            case ExtraScalarInt64PredicateOp::kNone:
                break;
        }
        return true;
    }

    bool
    test_double_predicate_(double value) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
        switch (filter.op) {
            case ExtraScalarInt64PredicateOp::kGreaterEqual:
                return value < filter.double_arg0;
            case ExtraScalarInt64PredicateOp::kGreaterThan:
                return value <= filter.double_arg0;
            case ExtraScalarInt64PredicateOp::kLessEqual:
                return value > filter.double_arg0;
            case ExtraScalarInt64PredicateOp::kLessThan:
                return value >= filter.double_arg0;
            case ExtraScalarInt64PredicateOp::kEqual:
                return value != filter.double_arg0;
            case ExtraScalarInt64PredicateOp::kNotEqual:
                return value == filter.double_arg0;
            case ExtraScalarInt64PredicateOp::kAddLessThan:
                return value + filter.double_arg0 >= filter.double_arg1;
            case ExtraScalarInt64PredicateOp::kSubLessThan:
                return value - filter.double_arg0 >= filter.double_arg1;
            case ExtraScalarInt64PredicateOp::kMulLessThan:
                return value * filter.double_arg0 >= filter.double_arg1;
            case ExtraScalarInt64PredicateOp::kDivLessThan:
                return filter.double_arg0 == 0.0 || value / filter.double_arg0 >= filter.double_arg1;
            case ExtraScalarInt64PredicateOp::kTerm:
                for (size_t i = 0; i < filter.double_term_count; ++i) {
                    if (value == filter.double_terms[i]) {
                        return false;
                    }
                }
                return true;
            case ExtraScalarInt64PredicateOp::kRange: {
                const bool lower_ok = filter.lower_inclusive ? value >= filter.double_arg0 : value > filter.double_arg0;
                const bool upper_ok = filter.upper_inclusive ? value <= filter.double_arg1 : value < filter.double_arg1;
                return !(lower_ok && upper_ok);
            }
            case ExtraScalarInt64PredicateOp::kModLessThan:
            case ExtraScalarInt64PredicateOp::kPrefixMatch:
            case ExtraScalarInt64PredicateOp::kPostfixMatch:
            case ExtraScalarInt64PredicateOp::kInnerMatch:
            case ExtraScalarInt64PredicateOp::kLikeMatch:
            case ExtraScalarInt64PredicateOp::kNone:
                break;
        }
        return true;
    }

    static size_t
    utf8_char_byte_len_(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) {
            return 1;
        }
        if (first_byte >= 0xC2 && first_byte <= 0xDF) {
            return 2;
        }
        if ((first_byte & 0xF0) == 0xE0) {
            return 3;
        }
        if (first_byte >= 0xF0 && first_byte <= 0xF7) {
            return 4;
        }
        return 1;
    }

    static size_t
    utf8_wildcard_char_byte_len_(const char* data, size_t remaining) {
        if (remaining == 0) {
            return 0;
        }
        const auto first_byte = static_cast<unsigned char>(data[0]);
        const auto char_len = utf8_char_byte_len_(first_byte);
        if (char_len == 1) {
            return first_byte <= 0x7F ? 1 : 0;
        }
        if (char_len > remaining) {
            return 0;
        }
        for (size_t i = 1; i < char_len; ++i) {
            if ((static_cast<unsigned char>(data[i]) & 0xC0) != 0x80) {
                return 0;
            }
        }
        return char_len;
    }

    bool
    matches_compiled_like_(std::string_view value) const {
        constexpr uint8_t kLiteral = 0;
        constexpr uint8_t kAnyOne = 1;
        constexpr uint8_t kAnyMany = 2;
        constexpr size_t kNoStar = std::numeric_limits<size_t>::max();

        const auto& filter = extra_scalar_int64_predicate_filter_;
        const auto& pattern = filter.like_pattern;
        if (pattern.token_count > 0 && (filter.string_arg0_data == nullptr || pattern.token_offsets == nullptr ||
                                        pattern.token_sizes == nullptr || pattern.token_types == nullptr)) {
            return false;
        }

        size_t token_idx = 0;
        size_t value_pos = 0;
        size_t star_token_idx = kNoStar;
        size_t star_value_pos = 0;
        while (value_pos < value.size()) {
            if (token_idx < pattern.token_count) {
                const auto token_type = pattern.token_types[token_idx];
                if (token_type == kAnyMany) {
                    if (token_idx + 1 == pattern.token_count) {
                        return true;
                    }
                    star_token_idx = token_idx++;
                    star_value_pos = value_pos;
                    continue;
                }
                if (token_type == kAnyOne) {
                    const auto char_len =
                        utf8_wildcard_char_byte_len_(value.data() + value_pos, value.size() - value_pos);
                    if (char_len > 0) {
                        value_pos += char_len;
                        ++token_idx;
                        continue;
                    }
                } else if (token_type == kLiteral) {
                    const auto offset = pattern.token_offsets[token_idx];
                    const auto size = pattern.token_sizes[token_idx];
                    if (offset <= filter.string_arg0_size && size <= filter.string_arg0_size - offset &&
                        size <= value.size() - value_pos &&
                        std::memcmp(value.data() + value_pos, filter.string_arg0_data + offset, size) == 0) {
                        value_pos += size;
                        ++token_idx;
                        continue;
                    }
                }
            }

            if (star_token_idx == kNoStar) {
                return false;
            }
            const auto char_len =
                utf8_wildcard_char_byte_len_(value.data() + star_value_pos, value.size() - star_value_pos);
            if (char_len == 0) {
                return false;
            }
            star_value_pos += char_len;
            value_pos = star_value_pos;
            token_idx = star_token_idx + 1;
        }
        while (token_idx < pattern.token_count && pattern.token_types[token_idx] == kAnyMany) {
            ++token_idx;
        }
        return token_idx == pattern.token_count;
    }

    bool
    test_string_predicate_(std::string_view value) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
        const std::string_view arg0(filter.string_arg0_data == nullptr ? "" : filter.string_arg0_data,
                                    filter.string_arg0_size);
        const std::string_view arg1(filter.string_arg1_data == nullptr ? "" : filter.string_arg1_data,
                                    filter.string_arg1_size);
        switch (filter.op) {
            case ExtraScalarInt64PredicateOp::kGreaterEqual:
                return value < arg0;
            case ExtraScalarInt64PredicateOp::kGreaterThan:
                return value <= arg0;
            case ExtraScalarInt64PredicateOp::kLessEqual:
                return value > arg0;
            case ExtraScalarInt64PredicateOp::kLessThan:
                return value >= arg0;
            case ExtraScalarInt64PredicateOp::kEqual:
                return value != arg0;
            case ExtraScalarInt64PredicateOp::kNotEqual:
                return value == arg0;
            case ExtraScalarInt64PredicateOp::kTerm: {
                if (!filter.string_terms_sorted) {
                    for (size_t i = 0; i < filter.string_term_count; ++i) {
                        const std::string_view term(filter.string_term_values[i], filter.string_term_sizes[i]);
                        if (value == term) {
                            return false;
                        }
                    }
                    return true;
                }
                size_t left = 0;
                size_t right = filter.string_term_count;
                while (left < right) {
                    const auto mid = left + (right - left) / 2;
                    const std::string_view term(filter.string_term_values[mid], filter.string_term_sizes[mid]);
                    const auto cmp = value.compare(term);
                    if (cmp == 0) {
                        return false;
                    }
                    if (cmp < 0) {
                        right = mid;
                    } else {
                        left = mid + 1;
                    }
                }
                return true;
            }
            case ExtraScalarInt64PredicateOp::kRange: {
                const bool lower_ok = filter.lower_inclusive ? value >= arg0 : value > arg0;
                const bool upper_ok = filter.upper_inclusive ? value <= arg1 : value < arg1;
                return !(lower_ok && upper_ok);
            }
            case ExtraScalarInt64PredicateOp::kPrefixMatch:
                return !value.starts_with(arg0);
            case ExtraScalarInt64PredicateOp::kPostfixMatch:
                return !value.ends_with(arg0);
            case ExtraScalarInt64PredicateOp::kInnerMatch:
                return value.find(arg0) == std::string_view::npos;
            case ExtraScalarInt64PredicateOp::kLikeMatch:
                return !matches_compiled_like_(value);
            case ExtraScalarInt64PredicateOp::kAddLessThan:
            case ExtraScalarInt64PredicateOp::kSubLessThan:
            case ExtraScalarInt64PredicateOp::kMulLessThan:
            case ExtraScalarInt64PredicateOp::kDivLessThan:
            case ExtraScalarInt64PredicateOp::kModLessThan:
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

    bool
    get_extra_scalar_float_predicate_value_(int64_t out_id, float* value) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
        if (value == nullptr || out_id < 0 || static_cast<size_t>(out_id) >= filter.row_count) {
            return false;
        }
        if (filter.row_float_values != nullptr) {
            *value = filter.row_float_values[out_id];
            return true;
        }
        if (filter.chunk_float_values != nullptr && filter.chunk_offsets != nullptr && filter.num_chunks > 0) {
            if (filter.num_chunks == 1) {
                *value = filter.chunk_float_values[0][out_id];
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
            *value = filter.chunk_float_values[chunk_idx][out_id - filter.chunk_offsets[chunk_idx]];
            return true;
        }
        return false;
    }

    bool
    get_extra_scalar_string_predicate_value_(int64_t out_id, std::string_view* value, bool* is_valid) const {
        const auto& filter = extra_scalar_int64_predicate_filter_;
        const auto& column = filter.string_column;
        if (value == nullptr || is_valid == nullptr || out_id < 0 || static_cast<size_t>(out_id) >= column.row_count ||
            column.chunk_bases == nullptr || column.chunk_value_offsets == nullptr ||
            column.chunk_row_counts == nullptr || column.chunk_row_offsets == nullptr || column.num_chunks == 0) {
            return false;
        }

        size_t chunk_idx = 0;
        size_t local_offset = static_cast<size_t>(out_id);
        if (column.num_chunks > 1) {
            if (column.uniform_chunk_rows > 0) {
                chunk_idx = local_offset / column.uniform_chunk_rows;
                if (chunk_idx >= column.num_chunks) {
                    return false;
                }
                local_offset -= chunk_idx * column.uniform_chunk_rows;
            } else {
                const auto* upper = std::upper_bound(column.chunk_row_offsets,
                                                     column.chunk_row_offsets + column.num_chunks + 1, out_id);
                if (upper == column.chunk_row_offsets) {
                    return false;
                }
                chunk_idx = static_cast<size_t>((upper - column.chunk_row_offsets) - 1);
                if (chunk_idx >= column.num_chunks) {
                    return false;
                }
                local_offset -= static_cast<size_t>(column.chunk_row_offsets[chunk_idx]);
            }
        }

        const auto* base = column.chunk_bases[chunk_idx];
        const auto* offsets = column.chunk_value_offsets[chunk_idx];
        const auto* valid_data = column.chunk_valid_data == nullptr ? nullptr : column.chunk_valid_data[chunk_idx];
        if (base == nullptr || offsets == nullptr || local_offset >= column.chunk_row_counts[chunk_idx]) {
            return false;
        }
        if (valid_data != nullptr && !valid_data[local_offset]) {
            *is_valid = false;
            *value = std::string_view();
            return true;
        }

        const auto begin = offsets[local_offset];
        const auto end = offsets[local_offset + 1];
        if (end < begin) {
            return false;
        }
        *is_valid = true;
        *value = std::string_view(base + begin, end - begin);
        return true;
    }
};
}  // namespace knowhere

#endif /* BITSET_H */
