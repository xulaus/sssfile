#include "catch.hpp"
#include <gsl/gsl>

#include "column_builder.h"

using namespace SSSFile;


TEST_CASE("Can convert column", "[]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;

    auto col = build_column_from_buffer("-01\n+ 2\ns-3", column_details);
    REQUIRE(col[0] == 1);
    REQUIRE(col[1] == 2);
    REQUIRE(col[2] ==-3);
}
