#include "catch.hpp"

#include "utf_conversion.h"

using namespace SSSFile;

TEST_CASE("UTF-8 to UTF-32 conversion", "[encoding]")
{
    const char *to_convert = "\x44\xC2\xA2\xE2\x82\xAC\xF0\x90\x8D\x88\x41";

    const int32_t expected[5] = {0x44, 0xA2, 0x20AC, 0x10348, 0x41};
    const size_t char_width[5] = {1, 2, 3, 4, 1};

    for (int i = 0, strpos = 0; i < 5; strpos += char_width[i], i++)
    {
        int32_t ret = 0;
        REQUIRE(utf8_to_uft32(to_convert, strpos, ret) == char_width[i]);
        REQUIRE(ret == expected[i]);
    }
}
