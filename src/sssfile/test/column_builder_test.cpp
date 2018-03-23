#include "catch.hpp"
#include <string_view>

#include "sssfile/column_builder.h"

using namespace SSSFile;

TEST_CASE("Can convert into an integer column", "[to_i]")
{
    auto column_details = sss_column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;
    column_details.type = sss_column_metadata::TYPE_INT32;

    auto column_text = "-01\n+ 2\ns-3\n";

    int col[3];
    REQUIRE(fill_column_from_cstr((void *) col, column_text, column_details) == SUCCESS);
    CHECK(col[0] == 1);
    CHECK(col[1] == 2);
    CHECK(col[2] == -3);
}

TEST_CASE("Fails gracefully given floats", "[to_i]")
{
    auto column_details = sss_column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;
    column_details.type = sss_column_metadata::TYPE_INT32;

    auto column_text = "3.14159265358979\n"
                       "            2E-1\n"
                       "              -3";

    int col[3];
    REQUIRE(fill_column_from_cstr((void *) col, column_text, column_details) == INVALID_NUMBER);
}

TEST_CASE("Integer conversion fails gracefully given nonsense", "[to_i]")
{
    auto column_details = sss_column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;
    column_details.type = sss_column_metadata::TYPE_INT32;

    auto column_text = "-01\n+ 2\ns-d\n";

    int col[3];
    REQUIRE(fill_column_from_cstr((void *) col, column_text, column_details) == INVALID_NUMBER);
}

TEST_CASE("Can convert into a double column", "[to_f]")
{
    auto column_details = sss_column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;
    column_details.type = sss_column_metadata::TYPE_DOUBLE;

    auto column_text = "3.14159265358979\n"
                       "            2E-1\n"
                       "              -3";

    double col[3];
    REQUIRE(fill_column_from_cstr((void *) col, column_text, column_details) == SUCCESS);
    CHECK(col[0] == 3.14159265358979);
    CHECK(col[1] == 0.2);
    CHECK(col[2] == -3);
}

TEST_CASE("Double Conversion fails gracefully given nonsense", "[to_f]")
{
    auto column_details = sss_column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;
    column_details.type = sss_column_metadata::TYPE_DOUBLE;

    auto column_text = "3.14159265358979\n"
                       "         HELLO  \n"
                       "              -3";

    double col[3];
    REQUIRE(fill_column_from_cstr((void *) col, column_text, column_details) == INVALID_NUMBER);
}
