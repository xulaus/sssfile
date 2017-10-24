#include <cstdint>
#include "catch.hpp"

#include "common.h"

using namespace SSSFile;

TEST_CASE("Can convert simple span to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "195" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 195);
}

TEST_CASE("Can convert single letter span to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "0" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0);
}

TEST_CASE("Can convert span with negative to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "-81" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == -81);
}

TEST_CASE("Can convert negative 0 successfully to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "-0" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0);
}

TEST_CASE("Can convert positive 0 successfully to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "+0" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0);
}

TEST_CASE("Can convert span with explicit positive to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "+1230" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 1230);
}

TEST_CASE("Can convert span padded with spaces to integer", "[string_view][to_i]")
{
    auto s = std::string_view { " 1230" };
    int ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 1230);
}

TEST_CASE("Cannot convert float string to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "195.0" };
    int ret = INT_MAX;
    REQUIRE_FALSE(to_numeric(s, ret));
    REQUIRE(ret == INT_MAX);
}

TEST_CASE("Cannot convert gibberish string to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "fdsas.j" };
    int ret = INT_MAX;
    REQUIRE_FALSE(to_numeric(s, ret));
    REQUIRE(ret == INT_MAX);
}

TEST_CASE("Cannot convert exponent string to integer", "[string_view][to_i]")
{
    auto s = std::string_view { "33e3" };
    int ret = INT_MAX;
    REQUIRE_FALSE(to_numeric(s, ret));
    REQUIRE(ret == INT_MAX);
}

TEST_CASE("Can convert simple span to float", "[string_view][to_f]")
{
    auto s = std::string_view { "193" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 193.0);
}

TEST_CASE("Can convert span with trailing '.' to float", "[string_view][to_f]")
{
    auto s = std::string_view { "5922." };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 5922.0);
}

TEST_CASE("Can convert span with fractional part to float", "[string_view][to_f]")
{
    auto s = std::string_view { "74.256" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 74.256);
}

TEST_CASE("Can convert span without whole part to float", "[string_view][to_f]")
{
    auto s = std::string_view { ".453" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0.453);
}

TEST_CASE("Can convert span with implicit positive matisa to float", "[string_view][to_f]")
{
    auto s = std::string_view { ".4E2" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == Approx(40.0));
}

TEST_CASE("Can convert span with explicit positive matisa to float", "[string_view][to_f]")
{
    auto s = std::string_view { "6.6E+2" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == Approx(660.0));
}

TEST_CASE("Can convert span with negative matisa to float", "[string_view][to_f]")
{
    auto s = std::string_view { "73e-3" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == Approx(0.073));
}

TEST_CASE("Can convert single letter span to float", "[string_view][to_f]")
{
    auto s = std::string_view { "0" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0);
}

TEST_CASE("Can convert span with negative to float", "[string_view][to_f]")
{
    auto s = std::string_view { "-81" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == -81);
}

TEST_CASE("Can convert negative 0 successfully to float", "[string_view][to_f]")
{
    auto s = std::string_view { "-0" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0);
}

TEST_CASE("Can convert positive 0 successfully to float", "[string_view][to_f]")
{
    auto s = std::string_view { "+0" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 0);
}

TEST_CASE("Can convert span with explicit positive to float", "[string_view][to_f]")
{
    auto s = std::string_view { "+1230" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 1230);
}

TEST_CASE("Can convert span with negative, padded with spaces to float", "[string_view][to_f]")
{
    auto s = std::string_view { " -1230" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == -1230);
}

TEST_CASE("Can convert span with positive, padded with spaces to float", "[string_view][to_f]")
{
    auto s = std::string_view { " +1230" };
    double ret = INT_MAX;
    REQUIRE(to_numeric(s, ret));
    REQUIRE(ret == 1230);
}

TEST_CASE("Cannot convert gibberish to float", "[string_view][to_f]")
{
    auto s = std::string_view { "asdfadsf" };
    double ret = INT_MAX;
    REQUIRE_FALSE(to_numeric(s, ret));
    REQUIRE(ret == INT_MAX);
}
