#include "catch.hpp"

#include "sss_string.h"

using namespace SSSFile;

TEST_CASE("Can convert simple span to integer", "[string_view  to_i]")
{
    auto s = std::string_view { "195" };
    REQUIRE(to_i(s) == 195);
}

TEST_CASE("Can convert single letter span to integer", "[string_view  to_i]")
{
    auto s = std::string_view { "0" };
    REQUIRE(to_i(s) == 0);
}

TEST_CASE("Can convert span with negative to integer", "[string_view  to_i]")
{
    auto s = std::string_view { "-81" };
    REQUIRE(to_i(s) == -81);
}

TEST_CASE("Can convert negative 0 successfully to integer", "[string_view  to_i]")
{
    auto s = std::string_view { "-0" };
    REQUIRE(to_i(s) == 0);
}

TEST_CASE("Can convert positive 0 successfully to integer", "[string_view  to_i]")
{
    auto s = std::string_view { "+0" };
    REQUIRE(to_i(s) == 0);
}

TEST_CASE("Can convert span with explicit positive to integer", "[string_view  to_i]")
{
    auto s = std::string_view { "+1230" };
    REQUIRE(to_i(s) == 1230);
}

TEST_CASE("Can convert span padded with spaces to integer", "[string_view  to_i]")
{
    auto s = std::string_view { " 1230" };
    REQUIRE(to_i(s) == 1230);
}

TEST_CASE("Can convert simple span to float", "[string_view  to_f]")
{
    auto s = std::string_view { "193" };
    REQUIRE(to_f(s) == Approx(193.0));
}

TEST_CASE("Can convert span with trailing '.' to float", "[string_view  to_f]")
{
    auto s = std::string_view { "5922." };
    REQUIRE(to_f(s) == Approx(5922.0));
}

TEST_CASE("Can convert span with fractional part to float", "[string_view  to_f]")
{
    auto s = std::string_view { "74.256" };
    REQUIRE(to_f(s) == Approx(74.256));
}

TEST_CASE("Can convert span without whole part to float", "[string_view  to_f]")
{
    auto s = std::string_view { ".453" };
    REQUIRE(to_f(s) == Approx(0.453));
}

TEST_CASE("Can convert span with implicit positive matisa to float", "[string_view  to_f]")
{
    auto s = std::string_view { ".4E2" };
    REQUIRE(to_f(s) == Approx(40.0));
}

TEST_CASE("Can convert span with explicit positive matisa to float", "[string_view  to_f]")
{
    auto s = std::string_view { ".4E+2" };
    REQUIRE(to_f(s) == Approx(40.0));
}

TEST_CASE("Can convert span with negative matisa to float", "[string_view  to_f]")
{
    auto s = std::string_view { "73e-3" };
    REQUIRE(to_f(s) == Approx(0.073));
}

TEST_CASE("Can convert single letter span to float", "[string_view  to_f]")
{
    auto s = std::string_view { "0" };
    REQUIRE(to_f(s) == Approx(0));
}

TEST_CASE("Can convert span with negative to float", "[string_view  to_f]")
{
    auto s = std::string_view { "-81" };
    REQUIRE(to_f(s) == Approx(-81));
}

TEST_CASE("Can convert negative 0 successfully to float", "[string_view  to_f]")
{
    auto s = std::string_view { "-0" };
    REQUIRE(to_f(s) == Approx(0));
}

TEST_CASE("Can convert positive 0 successfully to float", "[string_view  to_f]")
{
    auto s = std::string_view { "+0" };
    REQUIRE(to_f(s) == Approx(0));
}

TEST_CASE("Can convert span with explicit positive to float", "[string_view  to_f]")
{
    auto s = std::string_view { "+1230" };
    REQUIRE(to_f(s) == Approx(1230));
}

TEST_CASE("Can convert span with negative, padded with spaces to float", "[string_view  to_f]")
{
    auto s = std::string_view { " -1230" };
    REQUIRE(to_f(s) == Approx(-1230));
}

TEST_CASE("Can convert span with positive, padded with spaces to float", "[string_view  to_f]")
{
    auto s = std::string_view { " +1230" };
    REQUIRE(to_f(s) == Approx(1230));
}
