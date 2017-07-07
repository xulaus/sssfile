#include "catch.hpp"
#include <gsl/gsl>

#include "column_builder.h"

using namespace SSSFile;


TEST_CASE("Can convert column", "[]")
{
    auto col = build_column_from_buffer("01\n 2,-3",0,2,3);
    REQUIRE(col[0] == 1);
    REQUIRE(col[1] == 2);
    REQUIRE(col[2] ==-3);
}
