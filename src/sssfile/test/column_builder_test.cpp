#include "catch.hpp"
#include <gsl/gsl>

#include "column_builder.h"

using namespace SSSFile;


TEST_CASE("Can convert into an integer column", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;

    auto col = build_integer_column_from_buffer("-01\n+ 2\ns-3", column_details);
    REQUIRE(col[0] == 1);
    REQUIRE(col[1] == 2);
    REQUIRE(col[2] ==-3);
}


TEST_CASE("Can convert into a float column", "[to_f]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;

    auto col = build_float_column_from_buffer(
        "3.14159265358979 "
        "            2E-1 "
        "              -3",
        column_details
    );

    REQUIRE(col[0] == Approx(3.14159265358979));
    REQUIRE(col[1] == Approx(0.2));
    REQUIRE(col[2] == Approx(-3));
}
