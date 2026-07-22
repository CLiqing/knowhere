#include <array>
#include <cstdint>
#include <string_view>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"

namespace {

knowhere::BitsetView::ExtraScalarInt64PredicateFilter
MakeFilter(knowhere::BitsetView::ExtraScalarInt64PredicateOp op, std::string_view arg) {
    static const std::array<char, 4> chunk0 = {'a', 'a', 'b', 'b'};
    static const std::array<char, 10> chunk1 = {'c', 'c', 'c', 'd', 'd', 'b', 'b', 'e', 'e', 'e'};
    static const std::array<uint32_t, 3> offsets0 = {0, 2, 4};
    static const std::array<uint32_t, 5> offsets1 = {0, 3, 5, 7, 10};
    static const std::array<bool, 2> valid0 = {true, false};
    static const std::array<bool, 4> valid1 = {true, true, true, true};
    static const std::array<const char*, 2> bases = {chunk0.data(), chunk1.data()};
    static const std::array<const uint32_t*, 2> value_offsets = {offsets0.data(), offsets1.data()};
    static const std::array<const bool*, 2> valid_data = {valid0.data(), valid1.data()};
    static const std::array<size_t, 2> row_counts = {2, 4};
    static const std::array<int64_t, 3> row_offsets = {0, 2, 6};

    knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
    filter.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kString;
    filter.op = op;
    filter.row_count = 6;
    filter.string_arg0_data = arg.data();
    filter.string_arg0_size = static_cast<uint32_t>(arg.size());
    filter.string_column.chunk_bases = bases.data();
    filter.string_column.chunk_value_offsets = value_offsets.data();
    filter.string_column.chunk_valid_data = valid_data.data();
    filter.string_column.chunk_row_counts = row_counts.data();
    filter.string_column.chunk_row_offsets = row_offsets.data();
    filter.string_column.num_chunks = 2;
    filter.string_column.row_count = 6;
    return filter;
}

knowhere::BitsetView
MakeView(const knowhere::BitsetView::ExtraScalarInt64PredicateFilter& filter) {
    static const std::array<uint8_t, 1> bits = {0};
    knowhere::BitsetView view(bits.data(), 6);
    view.set_extra_scalar_int64_predicate_filter(filter, 0);
    return view;
}

}  // namespace

TEST_CASE("BitsetView reads nullable varchar from irregular raw chunks") {
    auto equal = MakeView(MakeFilter(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kEqual, "bb"));

    CHECK(equal.test(0));
    CHECK(equal.test(1));
    CHECK_FALSE(equal.test(4));

    auto not_equal = MakeView(MakeFilter(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kNotEqual, "zz"));
    CHECK_FALSE(not_equal.test(0));
    CHECK(not_equal.test(1));
}

TEST_CASE("BitsetView exposes special ID layout metadata") {
    static constexpr std::array<uint8_t, 1> bits = {0};
    static constexpr std::array<uint32_t, 3> out_ids = {2, 0, 1};

    knowhere::BitsetView ordinary(bits.data(), 3);
    CHECK_FALSE(ordinary.has_out_ids());
    CHECK(ordinary.id_offset() == 0);

    ordinary.set_out_ids(out_ids.data(), out_ids.size(), 0);
    CHECK(ordinary.has_out_ids());
    CHECK(ordinary.id_offset() == 0);

    ordinary.set_id_offset(17);
    CHECK(ordinary.id_offset() == 17);
}

TEST_CASE("BitsetView evaluates compiled varchar LIKE without reparsing") {
    static constexpr std::string_view pattern = "%b_";
    static constexpr std::array<uint32_t, 3> token_offsets = {0, 1, 2};
    static constexpr std::array<uint32_t, 3> token_sizes = {1, 1, 1};
    static constexpr std::array<uint8_t, 3> token_types = {2, 0, 1};

    auto filter = MakeFilter(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kLikeMatch, pattern);
    filter.like_pattern.token_offsets = token_offsets.data();
    filter.like_pattern.token_sizes = token_sizes.data();
    filter.like_pattern.token_types = token_types.data();
    filter.like_pattern.token_count = token_types.size();
    auto view = MakeView(filter);

    CHECK(view.test(0));
    CHECK(view.test(1));
    CHECK_FALSE(view.test(4));
    CHECK(view.test(5));
}

TEST_CASE("BitsetView compiled LIKE handles UTF-8, escaping, and empty patterns") {
    static constexpr std::array<char, 8> values = {'a', '\xE4', '\xB8', '\xAD', 'b', 'a', '_', 'b'};
    static constexpr std::array<uint32_t, 4> offsets = {0, 5, 8, 8};
    static constexpr std::array<const char*, 1> bases = {values.data()};
    static constexpr std::array<const uint32_t*, 1> value_offsets = {offsets.data()};
    static constexpr std::array<const bool*, 1> valid_data = {nullptr};
    static constexpr std::array<size_t, 1> row_counts = {3};
    static constexpr std::array<int64_t, 2> row_offsets = {0, 3};

    auto make_filter = [](std::string_view pattern) {
        knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
        filter.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kString;
        filter.op = knowhere::BitsetView::ExtraScalarInt64PredicateOp::kLikeMatch;
        filter.row_count = 3;
        filter.string_arg0_data = pattern.data();
        filter.string_arg0_size = static_cast<uint32_t>(pattern.size());
        filter.string_column.chunk_bases = bases.data();
        filter.string_column.chunk_value_offsets = value_offsets.data();
        filter.string_column.chunk_valid_data = valid_data.data();
        filter.string_column.chunk_row_counts = row_counts.data();
        filter.string_column.chunk_row_offsets = row_offsets.data();
        filter.string_column.num_chunks = 1;
        filter.string_column.row_count = 3;
        return filter;
    };

    static constexpr std::string_view wildcard_pattern = "a_b";
    static constexpr std::array<uint32_t, 3> wildcard_offsets = {0, 1, 2};
    static constexpr std::array<uint32_t, 3> wildcard_sizes = {1, 1, 1};
    static constexpr std::array<uint8_t, 3> wildcard_types = {0, 1, 0};
    auto wildcard_filter = make_filter(wildcard_pattern);
    wildcard_filter.like_pattern.token_offsets = wildcard_offsets.data();
    wildcard_filter.like_pattern.token_sizes = wildcard_sizes.data();
    wildcard_filter.like_pattern.token_types = wildcard_types.data();
    wildcard_filter.like_pattern.token_count = wildcard_types.size();
    auto wildcard_view = MakeView(wildcard_filter);
    CHECK_FALSE(wildcard_view.test(0));
    CHECK_FALSE(wildcard_view.test(1));
    CHECK(wildcard_view.test(2));

    static constexpr std::string_view escaped_pattern = "a\\_b";
    static constexpr std::array<uint32_t, 3> escaped_offsets = {0, 2, 3};
    static constexpr std::array<uint32_t, 3> escaped_sizes = {1, 1, 1};
    static constexpr std::array<uint8_t, 3> escaped_types = {0, 0, 0};
    auto escaped_filter = make_filter(escaped_pattern);
    escaped_filter.like_pattern.token_offsets = escaped_offsets.data();
    escaped_filter.like_pattern.token_sizes = escaped_sizes.data();
    escaped_filter.like_pattern.token_types = escaped_types.data();
    escaped_filter.like_pattern.token_count = escaped_types.size();
    auto escaped_view = MakeView(escaped_filter);
    CHECK(escaped_view.test(0));
    CHECK_FALSE(escaped_view.test(1));
    CHECK(escaped_view.test(2));

    auto empty_view = MakeView(make_filter(""));
    CHECK(empty_view.test(0));
    CHECK(empty_view.test(1));
    CHECK_FALSE(empty_view.test(2));
}

TEST_CASE("BitsetView raw string lookup uses uniform chunks") {
    static constexpr std::array<char, 4> chunk0 = {'a', 'a', 'b', 'b'};
    static constexpr std::array<char, 4> chunk1 = {'c', 'c', 'd', 'd'};
    static constexpr std::array<char, 2> chunk2 = {'e', 'e'};
    static constexpr std::array<uint32_t, 3> offsets0 = {0, 2, 4};
    static constexpr std::array<uint32_t, 3> offsets1 = {0, 2, 4};
    static constexpr std::array<uint32_t, 2> offsets2 = {0, 2};
    static constexpr std::array<const char*, 3> bases = {chunk0.data(), chunk1.data(), chunk2.data()};
    static constexpr std::array<const uint32_t*, 3> value_offsets = {offsets0.data(), offsets1.data(), offsets2.data()};
    static constexpr std::array<const bool*, 3> valid_data = {nullptr, nullptr, nullptr};
    static constexpr std::array<size_t, 3> row_counts = {2, 2, 1};
    static constexpr std::array<int64_t, 4> row_offsets = {0, 2, 4, 5};
    static constexpr std::string_view target = "dd";

    knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
    filter.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kString;
    filter.op = knowhere::BitsetView::ExtraScalarInt64PredicateOp::kEqual;
    filter.row_count = 5;
    filter.string_arg0_data = target.data();
    filter.string_arg0_size = static_cast<uint32_t>(target.size());
    filter.string_column.chunk_bases = bases.data();
    filter.string_column.chunk_value_offsets = value_offsets.data();
    filter.string_column.chunk_valid_data = valid_data.data();
    filter.string_column.chunk_row_counts = row_counts.data();
    filter.string_column.chunk_row_offsets = row_offsets.data();
    filter.string_column.num_chunks = bases.size();
    filter.string_column.row_count = 5;
    filter.string_column.uniform_chunk_rows = 2;

    auto view = MakeView(filter);
    CHECK(view.test(2));
    CHECK_FALSE(view.test(3));
    CHECK(view.test(4));
}
