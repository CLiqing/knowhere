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

#include <roaring/roaring.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace knowhere {
enum class BitsetPolarity : uint8_t {
    FilteredIds,
    ValidIds,
};

enum class FilterMapCapability : uint8_t {
    RandomMembership,
    EnumerateOnly,
};

using FilterMapTestFn = bool (*)(const void*, size_t);
using FilterMapReadUnsetFn = size_t (*)(const void*, size_t*, int32_t*, size_t);
using FilterMapGetUnsetSpanFn = bool (*)(const void*, const int32_t**, size_t*);
using FilterMapEnsureDenseFn = const uint8_t* (*)(const void*);

class BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits, size_t num_filtered_out_bits = 0, size_t id_offset = 0)
        : kind_(Kind::Dense),
          bits_(data),
          num_bits_(num_bits),
          num_filtered_out_bits_(num_filtered_out_bits),
          count_known_(num_filtered_out_bits != 0),
          id_offset_(id_offset) {
    }

    BitsetView(const roaring_bitmap_t* bitmap, size_t num_bits, size_t num_filtered_out_bits = 0, size_t id_offset = 0)
        : kind_(Kind::Roaring),
          roaring_(bitmap),
          num_bits_(num_bits),
          num_filtered_out_bits_(num_filtered_out_bits),
          count_known_(num_filtered_out_bits != 0),
          id_offset_(id_offset) {
    }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    BitsetView(const std::nullptr_t, size_t num_bits, size_t num_filtered_out_bits = 0, size_t id_offset = 0)
        : BitsetView(static_cast<const uint8_t*>(nullptr), num_bits, num_filtered_out_bits, id_offset) {
    }

    static BitsetView
    FromOwnedRoaring(std::shared_ptr<const roaring_bitmap_t> bitmap, size_t num_bits,
                     std::optional<size_t> num_filtered_out_bits = std::nullopt, size_t id_offset = 0) {
        BitsetView bitset(bitmap.get(), num_bits, 0, id_offset);
        bitset.owned_roaring_ = std::move(bitmap);
        if (num_filtered_out_bits.has_value()) {
            bitset.set_count(num_filtered_out_bits.value());
        }
        return bitset;
    }

    // Native scalar BitmapIndex output contains accepted IDs rather than the
    // historical excluded-ID convention.  Keep this factory separate from
    // legacy constructors so a Dense bitset cannot accidentally claim this
    // polarity.
    static BitsetView
    FromOwnedRoaringValid(std::shared_ptr<const roaring_bitmap_t> bitmap, size_t num_bits,
                          std::optional<size_t> num_filtered_out_bits = std::nullopt, size_t id_offset = 0) {
        BitsetView bitset(bitmap.get(), num_bits, 0, id_offset);
        bitset.polarity_ = BitsetPolarity::ValidIds;
        bitset.owned_roaring_ = std::move(bitmap);
        if (num_filtered_out_bits.has_value()) {
            bitset.set_count(num_filtered_out_bits.value());
        }
        return bitset;
    }

    // Sparse accepted IDs in producer order.  This is intentionally not a
    // membership bitset: Cardinal's explicit BF path consumes the span
    // directly, while other index paths must reject it.
    static BitsetView
    FromOwnedValidIdList(std::shared_ptr<const std::vector<int32_t>> ids, size_t num_bits,
                         std::optional<size_t> num_filtered_out_bits = std::nullopt) {
        if (ids == nullptr) {
            throw std::invalid_argument("valid-ID list owner must not be null");
        }
        BitsetView bitset;
        bitset.kind_ = Kind::ValidIdList;
        bitset.polarity_ = BitsetPolarity::ValidIds;
        bitset.valid_ids_ = ids->data();
        bitset.valid_ids_count_ = ids->size();
        bitset.num_bits_ = num_bits;
        bitset.owned_valid_ids_ = std::move(ids);
        if (num_filtered_out_bits.has_value()) {
            bitset.set_count(num_filtered_out_bits.value());
        }
        return bitset;
    }

    // Type-erased canonical filter owned by the caller. Consumers branch on
    // capability, never on Milvus' concrete list/Roaring/Dense backend.
    static BitsetView
    FromFilterMap(std::shared_ptr<const void> owner, const void* context, size_t num_bits, size_t num_filtered_out_bits,
                  FilterMapCapability capability, FilterMapTestFn test, FilterMapReadUnsetFn read_unset,
                  FilterMapGetUnsetSpanFn get_unset_span = nullptr, FilterMapEnsureDenseFn ensure_dense = nullptr) {
        if (owner == nullptr || context == nullptr || read_unset == nullptr) {
            throw std::invalid_argument("FilterMap owner, context and cursor callback must not be null");
        }
        if (capability == FilterMapCapability::RandomMembership && test == nullptr) {
            throw std::invalid_argument("RandomMembership FilterMap requires a test callback");
        }
        BitsetView bitset;
        bitset.kind_ = Kind::FilterMap;
        bitset.filter_map_owner_ = std::move(owner);
        bitset.filter_map_context_ = context;
        bitset.filter_map_capability_ = capability;
        bitset.filter_map_test_ = test;
        bitset.filter_map_read_unset_ = read_unset;
        bitset.filter_map_get_unset_span_ = get_unset_span;
        bitset.filter_map_ensure_dense_ = ensure_dense;
        bitset.num_bits_ = num_bits;
        bitset.set_count(num_filtered_out_bits);
        return bitset;
    }

    static BitsetView
    FromFrozenRoaring(std::shared_ptr<const void> data_owner, const void* data, size_t byte_size, size_t num_bits,
                      std::optional<size_t> num_filtered_out_bits = std::nullopt, size_t id_offset = 0) {
        if (data == nullptr) {
            throw std::invalid_argument("frozen Roaring data must not be null");
        }
        if (data_owner == nullptr) {
            throw std::invalid_argument("frozen Roaring data must have an owner");
        }
        if (reinterpret_cast<uintptr_t>(data) % 32 != 0) {
            throw std::invalid_argument("frozen Roaring data must be 32-byte aligned");
        }
        const auto* bitmap = roaring_bitmap_frozen_view(static_cast<const char*>(data), byte_size);
        if (bitmap == nullptr) {
            throw std::invalid_argument("invalid frozen Roaring data");
        }
        BitsetView bitset(bitmap, num_bits, 0, id_offset);
        bitset.roaring_backing_owner_ = std::move(data_owner);
        bitset.owned_roaring_ = std::shared_ptr<const roaring_bitmap_t>(
            bitmap, [](const roaring_bitmap_t* p) { roaring_bitmap_free(const_cast<roaring_bitmap_t*>(p)); });
        if (num_filtered_out_bits.has_value()) {
            bitset.set_count(num_filtered_out_bits.value());
        }
        return bitset;
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

    bool
    has_count() const {
        return count_known_;
    }

    void
    set_count(size_t num_filtered_out_bits) {
        if (out_ids_ != nullptr) {
            num_filtered_out_ids_ = num_filtered_out_bits;
        } else {
            num_filtered_out_bits_ = num_filtered_out_bits;
        }
        count_known_ = true;
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
    is_dense() const {
        return kind_ == Kind::Dense;
    }

    bool
    is_roaring() const {
        return kind_ == Kind::Roaring;
    }

    bool
    is_roaring_valid() const {
        return is_roaring() && polarity_ == BitsetPolarity::ValidIds;
    }

    bool
    is_valid_id_list() const {
        return kind_ == Kind::ValidIdList;
    }

    bool
    is_filter_map() const {
        return kind_ == Kind::FilterMap;
    }

    FilterMapCapability
    filter_map_capability() const {
        return filter_map_capability_;
    }

    size_t
    read_filter_map_unset(size_t& cursor, std::span<int32_t> output) const {
        if (!is_filter_map()) {
            throw std::logic_error("unset cursor requested from a non-FilterMap bitset");
        }
        return filter_map_read_unset_(filter_map_context_, &cursor, output.data(), output.size());
    }

    std::optional<std::span<const int32_t>>
    filter_map_unset_span() const {
        if (!is_filter_map() || filter_map_get_unset_span_ == nullptr) {
            return std::nullopt;
        }
        const int32_t* data = nullptr;
        size_t size = 0;
        if (!filter_map_get_unset_span_(filter_map_context_, &data, &size)) {
            return std::nullopt;
        }
        if (data == nullptr && size != 0) {
            throw std::logic_error("FilterMap returned a null non-empty unset-ID span");
        }
        return std::span<const int32_t>(data, size);
    }

    void
    ensure_dense() {
        if (!is_filter_map()) {
            return;
        }
        if (filter_map_ensure_dense_ == nullptr) {
            throw std::logic_error("FilterMap does not provide Dense materialization");
        }
        bits_ = filter_map_ensure_dense_(filter_map_context_);
        if (bits_ == nullptr && num_bits_ != 0) {
            throw std::logic_error("FilterMap Dense materialization returned null storage");
        }
        kind_ = Kind::Dense;
        filter_map_capability_ = FilterMapCapability::RandomMembership;
    }

    std::span<const int32_t>
    valid_ids() const {
        return {valid_ids_, valid_ids_count_};
    }

    BitsetPolarity
    polarity() const {
        return polarity_;
    }

    const roaring_bitmap_t*
    roaring() const {
        return roaring_;
    }

    bool
    has_valid_storage() const {
        return num_bits_ == 0 ||
               (kind_ == Kind::Dense         ? bits_ != nullptr
                : kind_ == Kind::Roaring     ? roaring_ != nullptr
                : kind_ == Kind::ValidIdList ? valid_ids_ != nullptr || valid_ids_count_ == 0
                                             : filter_map_owner_ != nullptr && filter_map_context_ != nullptr &&
                                                   filter_map_read_unset_ != nullptr);
    }

    size_t
    id_offset() const {
        return id_offset_;
    }

    bool
    can_iterate_roaring_without_mapping() const {
        return kind_ == Kind::Roaring && out_ids_ == nullptr;
    }

    std::vector<uint8_t>
    ToDense() const {
        if (kind_ == Kind::FilterMap && filter_map_capability_ == FilterMapCapability::EnumerateOnly) {
            std::vector<uint8_t> dense(byte_size(), 0xff);
            if (!dense.empty() && (num_bits_ & 7) != 0) {
                dense.back() &= static_cast<uint8_t>((1u << (num_bits_ & 7)) - 1);
            }
            size_t cursor = 0;
            std::array<int32_t, 256> ids{};
            while (const auto n = read_filter_map_unset(cursor, ids)) {
                for (size_t i = 0; i < n; ++i) {
                    const auto id = static_cast<size_t>(ids[i]);
                    if (id >= num_bits_) {
                        throw std::out_of_range("FilterMap cursor returned an ID outside its universe");
                    }
                    dense[id >> 3] &= static_cast<uint8_t>(~(1u << (id & 7)));
                }
            }
            return dense;
        }
        std::vector<uint8_t> dense(byte_size(), 0);
        for (size_t i = 0; i < num_bits_; ++i) {
            if (test(i)) {
                dense[i >> 3] |= 0x1 << (i & 0x7);
            }
        }
        return dense;
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
        count_known_ = true;
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

    // if the test succeeds, then the index should be skipped during search; otherwise, it should be included.
    bool
    test(int64_t index) const {
        int64_t out_id = index + id_offset_;
        if (out_ids_ != nullptr) {
            out_id = out_ids_[out_id];
        }
        // when index is larger than the max_offset, ignore it
        if (out_id >= static_cast<int64_t>(num_bits_)) {
            return true;
        }
        if (kind_ == Kind::Roaring) {
            const bool contains = roaring_bitmap_contains(roaring_, static_cast<uint32_t>(out_id));
            return polarity_ == BitsetPolarity::FilteredIds ? contains : !contains;
        }
        if (kind_ == Kind::FilterMap) {
            if (filter_map_test_ == nullptr) {
                throw std::logic_error("EnumerateOnly FilterMap does not support random membership");
            }
            return filter_map_test_(filter_map_context_, static_cast<size_t>(out_id));
        }
        return bits_[out_id >> 3] & (0x1 << (out_id & 0x7));
    }
    // return the filtered ratio. if with id mapping, calculated by internal_ids rather than bits.
    float
    filter_ratio() const {
        return empty() ? 0.0f : ((float)count() / size());
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
        if (kind_ == Kind::Roaring) {
            const auto cardinality = roaring_bitmap_get_cardinality(roaring_);
            return polarity_ == BitsetPolarity::FilteredIds ? cardinality : num_bits_ - cardinality;
        }
        if (kind_ == Kind::FilterMap) {
            return num_filtered_out_bits_;
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
        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return num_internal_ids_;
        }
        if (kind_ == Kind::FilterMap && filter_map_capability_ == FilterMapCapability::EnumerateOnly) {
            size_t cursor = 0;
            int32_t id = 0;
            return read_filter_map_unset(cursor, std::span<int32_t>(&id, 1)) == 0 ? num_bits_ : static_cast<size_t>(id);
        }
        if (kind_ == Kind::Roaring) {
            for (size_t i = 0; i < num_bits_; i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return num_bits_;
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
    enum class Kind { Dense, Roaring, ValidIdList, FilterMap };

    Kind kind_ = Kind::Dense;
    BitsetPolarity polarity_ = BitsetPolarity::FilteredIds;
    const uint8_t* bits_ = nullptr;
    const roaring_bitmap_t* roaring_ = nullptr;
    const int32_t* valid_ids_ = nullptr;
    size_t valid_ids_count_ = 0;
    // Members are destroyed in reverse declaration order. Keep the frozen backing owner alive until after the
    // CRoaring frozen view stored in owned_roaring_ has been freed.
    std::shared_ptr<const void> roaring_backing_owner_;
    std::shared_ptr<const roaring_bitmap_t> owned_roaring_;
    std::shared_ptr<const std::vector<int32_t>> owned_valid_ids_;
    std::shared_ptr<const void> filter_map_owner_;
    const void* filter_map_context_ = nullptr;
    FilterMapCapability filter_map_capability_ = FilterMapCapability::RandomMembership;
    FilterMapTestFn filter_map_test_ = nullptr;
    FilterMapReadUnsetFn filter_map_read_unset_ = nullptr;
    FilterMapGetUnsetSpanFn filter_map_get_unset_span_ = nullptr;
    FilterMapEnsureDenseFn filter_map_ensure_dense_ = nullptr;
    size_t num_bits_ = 0;
    size_t num_filtered_out_bits_ = 0;
    bool count_known_ = false;

    // optional. many indexes will share one bitset, requiring offset to distinguish between them.
    //  like multi-chunk brute-force in /src/common/comp/brute_force.cc, or mv-only in /src/index/hnsw/faiss_hnsw.cc
    size_t id_offset_ = 0;  // offset of the internal ids

    // optional. bitset supports id mapping.
    // Even allows multiple ids to map to the same bit, so the number of internal ids and bits may be not equal.
    const uint32_t* out_ids_ = nullptr;
    size_t num_internal_ids_ = 0;
    size_t num_filtered_out_ids_ = 0;
};
}  // namespace knowhere

#endif /* BITSET_H */
