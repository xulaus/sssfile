#include <string_view>
#include "catch.hpp"

#include "sssfile/column_builder.h"

using namespace SSSFile;


TEST_CASE("Can convert into an integer column", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;

    auto column_text = std::string_view { "-01\n+ 2\ns-3\n" };

    auto col = build_integer_column_from_buffer(column_text, column_details);
    REQUIRE(col != nullptr);
    if(col)
    {
        CHECK(col[0] ==  1);
        CHECK(col[1] ==  2);
        CHECK(col[2] == -3);
    }
}

TEST_CASE("Fails gracefully given floats", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;

    auto column_text = std::string_view {
        "3.14159265358979\n"
        "            2E-1\n"
        "              -3"
    };

    auto col = build_integer_column_from_buffer(column_text, column_details);
    REQUIRE(col == nullptr);
}

TEST_CASE("Integer conversion fails gracefully given nonsense", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;

    auto column_text = std::string_view { "-01\n+ 2\ns-d\n" };

    auto col = build_integer_column_from_buffer(column_text, column_details);
    REQUIRE(col == nullptr);
}



TEST_CASE("Can convert into a double column", "[to_f]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;

    auto column_text = std::string_view {
        "3.14159265358979\n"
        "            2E-1\n"
        "              -3"
    };

    auto col = build_float_column_from_buffer(column_text, column_details);
    REQUIRE(col != nullptr);
    if(col)
    {
        CHECK(col[0] == 3.14159265358979);
        CHECK(col[1] == 0.2);
        CHECK(col[2] == -3);
    }
}

TEST_CASE("Double Conversion fails gracefully given nonsense", "[to_f]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;

    auto column_text = std::string_view {
        "3.14159265358979\n"
        "         HELLO  \n"
        "              -3"
    };

    auto col = build_float_column_from_buffer(column_text, column_details);
    REQUIRE(col == nullptr);
}
