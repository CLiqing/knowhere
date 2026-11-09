// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace knowhere {

// Positional batch bits. Small graph batches own one inline word; larger
// batches own their storage. No returned view borrows a worker's next scratch.
class FilterBatchResult {
 public:
    explicit FilterBatchResult(size_t count = 0, bool value = false) : count_(count) {
        if (count > 64) {
            words_.assign(WordCount(count), value ? ~uint64_t{0} : 0);
        } else {
            inline_word_ = value ? TailMask(count) : 0;
        }
        if (value && count != 0) {
            data()[WordCount(count) - 1] &= TailMask(count);
        }
    }

    static size_t
    WordCount(size_t count) noexcept {
        return count / 64 + (count % 64 != 0);
    }
    static uint64_t
    TailMask(size_t count) noexcept {
        return count == 0 ? 0 : count % 64 == 0 ? ~uint64_t{0} : (uint64_t{1} << (count % 64)) - 1;
    }
    static FilterBatchResult
    FromWord(size_t count, uint64_t word) {
        if (count > 64) {
            throw std::invalid_argument("word result requires at most 64 positions");
        }
        FilterBatchResult result(count);
        result.inline_word_ = word & TailMask(count);
        return result;
    }
    size_t
    size() const noexcept {
        return count_;
    }
    uint64_t*
    data() noexcept {
        return count_ <= 64 ? &inline_word_ : words_.data();
    }
    const uint64_t*
    data() const noexcept {
        return count_ <= 64 ? &inline_word_ : words_.data();
    }
    bool
    operator[](size_t index) const noexcept {
        return (data()[index / 64] >> (index % 64)) & 1;
    }
    void
    set(size_t index, bool value = true) noexcept {
        auto& word = data()[index / 64];
        const auto bit = uint64_t{1} << (index % 64);
        if (value)
            word |= bit;
        else
            word &= ~bit;
    }
    void
    flip() noexcept {
        for (size_t i = 0; i < WordCount(count_); ++i) data()[i] = ~data()[i];
        if (count_ != 0)
            data()[WordCount(count_) - 1] &= TailMask(count_);
    }
    bool
    all() const noexcept {
        for (size_t i = 0; i < WordCount(count_); ++i) {
            const auto mask = i + 1 == WordCount(count_) ? TailMask(count_) : ~uint64_t{0};
            if (data()[i] != mask)
                return false;
        }
        return true;
    }

 private:
    size_t count_ = 0;
    uint64_t inline_word_ = 0;
    std::vector<uint64_t> words_;
};

}  // namespace knowhere
