#include "catch.hpp"
#include <gsl/gsl>

#include "sss_string.h"

using namespace SSSFile;

TEST_CASE("Can convert simple span to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({'1', '9', '5'});
    REQUIRE(to_i(s) == 195);
}

TEST_CASE("Can convert single letter span to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({'0'});
    REQUIRE(to_i(s) == 0);
}

TEST_CASE("Can convert span with negative to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({'-', '8', '1'});
    REQUIRE(to_i(s) == -81);
}

TEST_CASE("Can convert negative 0 successfully to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({'-', '0'});
    REQUIRE(to_i(s) == 0);
}

TEST_CASE("Can convert positive 0 successfully to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({'+', '0'});
    REQUIRE(to_i(s) == 0);
}

TEST_CASE("Can convert span with explicit positive to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({'+', '1', '2', '3', '0'});
    REQUIRE(to_i(s) == 1230);
}

TEST_CASE("Can convert span padded with spaces to integer", "[gsl_span to_i]")
{
    auto s = gsl::span<const char>({' ', '1', '2', '3', '0'});
    REQUIRE(to_i(s) == 1230);
}

TEST_CASE("Can convert simple span to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'1', '9', '3'});
    REQUIRE(to_f(s) == Approx(193.0));
}

TEST_CASE("Can convert span with trailing '.' to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'5', '9', '2', '2', '.'});
    REQUIRE(to_f(s) == Approx(5922.0));
}

TEST_CASE("Can convert span with fractional part to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'7', '4', '.', '2', '5', '6'});
    REQUIRE(to_f(s) == Approx(74.256));
}

TEST_CASE("Can convert span without whole part to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'.', '4', '5', '3'});
    REQUIRE(to_f(s) == Approx(0.453));
}

TEST_CASE("Can convert span with implicit positive matisa to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'.', '4', 'E', '2'});
    REQUIRE(to_f(s) == Approx(40.0));
}

TEST_CASE("Can convert span with explicit positive matisa to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'.', '4', 'E', '2'});
    REQUIRE(to_f(s) == Approx(40.0));
}

TEST_CASE("Can convert span with negative matisa to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'7', '3', 'e', '-', '3'});
    REQUIRE(to_f(s) == Approx(0.073));
}

TEST_CASE("Can convert single letter span to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'0'});
    REQUIRE(to_f(s) == Approx(0));
}

TEST_CASE("Can convert span with negative to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'-', '8', '1'});
    REQUIRE(to_f(s) == Approx(-81));
}

TEST_CASE("Can convert negative 0 successfully to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'-', '0'});
    REQUIRE(to_f(s) == Approx(0));
}

TEST_CASE("Can convert positive 0 successfully to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'+', '0'});
    REQUIRE(to_f(s) == Approx(0));
}

TEST_CASE("Can convert span with explicit positive to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({'+', '1', '2', '3', '0'});
    REQUIRE(to_f(s) == Approx(1230));
}

TEST_CASE("Can convert span padded with spaces to float", "[gsl_span to_f]")
{
    auto s = gsl::span<const char>({' ', '1', '2', '3', '0'});
    REQUIRE(to_f(s) == Approx(1230));
}
