#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"

#ifdef KNOWHERE_WITH_CARDINAL
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"
#endif

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

#ifdef KNOWHERE_WITH_CARDINAL
TEST_CASE("Cardinal HNSW varchar batch search matches scalar end to end") {
    const auto env_int = [](const char* name, int64_t fallback) {
        const char* value = std::getenv(name);
        return value == nullptr ? fallback : std::max<int64_t>(1, std::strtoll(value, nullptr, 10));
    };
    const int64_t rows = env_int("VBP_P4_ROWS", 512);
    const int64_t queries_count = env_int("VBP_P4_QUERIES", 8);
    const int64_t dim = env_int("VBP_P4_DIM", 32);
    const int64_t topk = env_int("VBP_P4_TOPK", 20);
    const int64_t ef = env_int("VBP_P4_EF", 64);
    const int64_t graph_degree = env_int("VBP_P4_GRAPH_DEGREE", 32);
    const uint32_t string_length = static_cast<uint32_t>(env_int("VBP_P4_STRING_LENGTH", 32));
    const std::string target(string_length, 'm');

    INFO("rows=" << rows << " nq=" << queries_count << " dim=" << dim << " topk=" << topk << " ef=" << ef
                 << " graph_degree=" << graph_degree << " string_length=" << string_length);

    auto train = GenDataSet(rows, dim, 20260723);
    auto queries = GenDataSet(queries_count, dim, 20260724);

    knowhere::Json build_config;
    build_config[knowhere::meta::DIM] = dim;
    build_config[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    build_config[knowhere::indexparam::HNSW_M] = graph_degree;
    build_config[knowhere::indexparam::EFCONSTRUCTION] = 120;
    build_config["index_algo"] = "GRAPH";

    const auto version = std::max(knowhere::Version::GetCurrentVersion().VersionNumber(), 9);
    auto index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(index.Build(train, build_config) == knowhere::Status::success);

    std::vector<char> values(static_cast<size_t>(rows) * string_length);
    std::vector<uint32_t> offsets(static_cast<size_t>(rows) + 1);
    for (int64_t row = 0; row < rows; ++row) {
        offsets[static_cast<size_t>(row)] = static_cast<uint32_t>(row) * string_length;
        const char fill = row % 2 == 0 ? 'm' : 'x';
        std::fill_n(values.data() + static_cast<size_t>(row) * string_length, string_length, fill);
    }
    offsets.back() = static_cast<uint32_t>(values.size());

    const std::array<const char*, 1> chunk_bases = {values.data()};
    const std::array<const uint32_t*, 1> chunk_value_offsets = {offsets.data()};
    const std::array<const bool*, 1> chunk_valid_data = {nullptr};
    const std::array<size_t, 1> chunk_row_counts = {static_cast<size_t>(rows)};
    const std::array<int64_t, 2> chunk_row_offsets = {0, rows};
    std::vector<uint8_t> bits((static_cast<size_t>(rows) + 7) / 8, 0);

    knowhere::BitsetView::ExtraScalarInt64PredicateFilter predicate;
    predicate.value_type = knowhere::BitsetView::ExtraScalarPredicateValueType::kString;
    predicate.op = knowhere::BitsetView::ExtraScalarInt64PredicateOp::kEqual;
    predicate.row_count = rows;
    predicate.string_arg0_data = target.data();
    predicate.string_arg0_size = static_cast<uint32_t>(target.size());
    predicate.string_column.chunk_bases = chunk_bases.data();
    predicate.string_column.chunk_value_offsets = chunk_value_offsets.data();
    predicate.string_column.chunk_valid_data = chunk_valid_data.data();
    predicate.string_column.chunk_row_counts = chunk_row_counts.data();
    predicate.string_column.chunk_row_offsets = chunk_row_offsets.data();
    predicate.string_column.num_chunks = 1;
    predicate.string_column.row_count = rows;

    knowhere::BitsetView filter(bits.data(), rows);
    filter.set_extra_scalar_int64_predicate_filter(predicate, rows / 2);

    knowhere::Json scalar_config = build_config;
    scalar_config[knowhere::meta::TOPK] = topk;
    scalar_config[knowhere::indexparam::EF] = ef;
    scalar_config["level"] = 1;
    scalar_config["switch_ivf_ratio"] = 1.0;
    scalar_config["enable_batch_4"] = false;
    scalar_config["downpush_debug_counter"] = true;
    scalar_config["experimental_filter_batch_observe"] = true;
    scalar_config["experimental_filter_batch_observe_width"] = 16;
    scalar_config["experimental_filter_batch_mode"] = "off";

    auto reference_config = scalar_config;
    reference_config["experimental_filter_batch_mode"] = "force";
    reference_config["experimental_filter_batch_width"] = 16;
    reference_config["experimental_filter_batch_min_fill"] = 1;
    reference_config["experimental_filter_batch_tail_mode"] = "batch";
    reference_config["experimental_filter_batch_prefetch"] = false;
    reference_config["experimental_filter_batch_kernel"] = "reference";

    auto portable_config = reference_config;
    portable_config["experimental_filter_batch_prefetch"] = true;
    portable_config["experimental_filter_batch_kernel"] = "portable";

    const auto scalar = index.Search(queries, scalar_config, filter);
    const auto reference = index.Search(queries, reference_config, filter);
    const auto portable = index.Search(queries, portable_config, filter);
    REQUIRE(scalar.has_value());
    REQUIRE(reference.has_value());
    REQUIRE(portable.has_value());

    const auto require_same_result = [](const knowhere::DataSet& expected, const knowhere::DataSet& actual) {
        REQUIRE(actual.GetRows() == expected.GetRows());
        REQUIRE(actual.GetDim() == expected.GetDim());
        const auto count = static_cast<size_t>(expected.GetRows() * expected.GetDim());
        for (size_t i = 0; i < count; ++i) {
            REQUIRE(actual.GetIds()[i] == expected.GetIds()[i]);
            REQUIRE(actual.GetDistance()[i] == expected.GetDistance()[i]);
        }
    };

    require_same_result(*scalar.value(), *reference.value());
    require_same_result(*scalar.value(), *portable.value());

    const auto result_count = static_cast<size_t>(scalar.value()->GetRows() * scalar.value()->GetDim());
    for (size_t i = 0; i < result_count; ++i) {
        REQUIRE(scalar.value()->GetIds()[i] >= 0);
        REQUIRE(scalar.value()->GetIds()[i] % 2 == 0);
    }

}
#endif
