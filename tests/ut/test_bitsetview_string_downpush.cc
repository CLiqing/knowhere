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

TEST_CASE("BitsetView evaluates STL_SORT dictionary IDs for EQ and NE") {
    static constexpr std::array<int32_t, 6> row_ids = {2, 0, -1, 1, 2, 0};
    static constexpr std::array<uint8_t, 1> bits = {0};

    auto make_view = [&](knowhere::BitsetView::ExtraScalarInt64PredicateOp op, int32_t target_id, bool target_found) {
        knowhere::BitsetView view(bits.data(), row_ids.size());
        knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
        filter.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kDictionaryId;
        filter.op = op;
        filter.row_dictionary_ids = row_ids.data();
        filter.row_count = row_ids.size();
        filter.target_dictionary_id = target_id;
        filter.target_dictionary_id_found = target_found;
        view.set_extra_scalar_int64_predicate_filter(filter, 0);
        return view;
    };

    auto equal = make_view(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kEqual, 2, true);
    CHECK_FALSE(equal.test(0));
    CHECK(equal.test(1));
    CHECK(equal.test(2));
    CHECK_FALSE(equal.test(4));

    auto equal_missing = make_view(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kEqual, -1, false);
    CHECK(equal_missing.test(0));
    CHECK(equal_missing.test(5));

    auto not_equal = make_view(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kNotEqual, 2, true);
    CHECK(not_equal.test(0));
    CHECK_FALSE(not_equal.test(1));
    CHECK(not_equal.test(2));
    CHECK_FALSE(not_equal.test(3));

    auto not_equal_missing = make_view(knowhere::BitsetView::ExtraScalarInt64PredicateOp::kNotEqual, -1, false);
    CHECK_FALSE(not_equal_missing.test(0));
    CHECK(not_equal_missing.test(2));
    CHECK_FALSE(not_equal_missing.test(5));
}

TEST_CASE("BitsetView carries row count without an all-visible base bitmap") {
    static constexpr std::array<int64_t, 4> values = {10, 20, 30, 40};

    knowhere::BitsetView view;
    knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
    filter.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kInt64;
    filter.op = knowhere::BitsetView::ExtraScalarInt64PredicateOp::kGreaterEqual;
    filter.row_values = values.data();
    filter.row_count = values.size();
    filter.arg0 = 25;
    view.set_extra_scalar_int64_predicate_filter(filter, 1);

    CHECK(view.data() == nullptr);
    CHECK(view.size() == values.size());
    CHECK(view.count() == 0);
    CHECK(view.estimated_count() == 1);
    CHECK(view.filtered_out_count_for_index_search() == 1);
    CHECK(view.test(0));
    CHECK(view.test(1));
    CHECK_FALSE(view.test(2));
    CHECK_FALSE(view.test(3));
}

TEST_CASE("BitsetView maps physical vector rows to logical scalar rows lazily") {
    static constexpr std::array<int64_t, 4> values = {10, 20, 30, 40};
    static constexpr std::array<int64_t, 3> physical_to_logical = {2, 0, 3};
    // Physical row 1 is excluded by the base bitmap. The mapper must affect
    // only scalar lookup, not base bitmap addressing.
    static constexpr std::array<uint8_t, 1> bits = {0b00000010};

    auto map_row = [](const void* context, int64_t physical) -> int64_t {
        const auto* mapping = static_cast<const std::array<int64_t, 3>*>(context);
        return physical >= 0 && static_cast<size_t>(physical) < mapping->size() ? (*mapping)[physical] : -1;
    };

    knowhere::BitsetView view(bits.data(), physical_to_logical.size());
    knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
    filter.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kInt64;
    filter.op = knowhere::BitsetView::ExtraScalarInt64PredicateOp::kGreaterEqual;
    filter.row_values = values.data();
    filter.row_count = values.size();
    filter.arg0 = 25;
    filter.scalar_row_id_mapper_context = &physical_to_logical;
    filter.scalar_row_id_mapper = map_row;
    view.set_extra_scalar_int64_predicate_filter(filter, 1);

    CHECK_FALSE(view.test(0));
    CHECK(view.test(1));
    CHECK_FALSE(view.test(2));
}
