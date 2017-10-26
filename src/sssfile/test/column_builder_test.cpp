#include "catch.hpp"
#include <string_view>

#include "sssfile/column_builder.h"

using namespace SSSFile;

TEST_CASE("Can convert into an integer column", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;

    auto column_text = std::string_view{"-01\n+ 2\ns-3\n"};

    auto col = build_integer_column_from_buffer(column_text, column_details);
    REQUIRE(col != nullptr);
    if (col)
    {
        CHECK(col[0] == 1);
        CHECK(col[1] == 2);
        CHECK(col[2] == -3);
    }
}

TEST_CASE("Fails gracefully given floats", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;

    auto column_text = std::string_view{"3.14159265358979\n"
                                        "            2E-1\n"
                                        "              -3"};

    auto col = build_integer_column_from_buffer(column_text, column_details);
    REQUIRE(col == nullptr);
}

TEST_CASE("Integer conversion fails gracefully given nonsense", "[to_i]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 4;
    column_details.size = 2;
    column_details.offset = 1;

    auto column_text = std::string_view{"-01\n+ 2\ns-d\n"};

    auto col = build_integer_column_from_buffer(column_text, column_details);
    REQUIRE(col == nullptr);
}

TEST_CASE("Can convert into a double column", "[to_f]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 17;
    column_details.size = 16;
    column_details.offset = 0;

    auto column_text = std::string_view{"3.14159265358979\n"
                                        "            2E-1\n"
                                        "              -3"};

    auto col = build_float_column_from_buffer(column_text, column_details);
    REQUIRE(col != nullptr);
    if (col)
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

    auto column_text = std::string_view{"3.14159265358979\n"
                                        "         HELLO  \n"
                                        "              -3"};

    auto col = build_float_column_from_buffer(column_text, column_details);
    REQUIRE(col == nullptr);
}


// HACK: Not appropriate here
TEST_CASE("UTF-8 to UTF-32 Column", "[encoding]")
{
    auto column_details = SSSFile::column_metadata{};
    column_details.line_length = 5;
    column_details.size = 4;
    column_details.offset = 0;
    column_details.type = column_metadata::TYPE_UTF32;

    auto column_text = std::string_view { "\x44   \n"
                                          "\xC2\xA2  \n"
                                          "\xE2\x82\xAC \n"
                                          "\xF0\x90\x8D\x88\n"
                                          "\x41\xc2\xa2 \n"};
    int32_t expected[20] = {0x44   , 0x20, 0x20, 0x20,
                            0xA2   , 0x20, 0x20, 0x00,
                            0x20AC , 0x20, 0x00, 0x00,
                            0x10348, 0x00, 0x00, 0x00,
                            0x41   , 0xA2, 0x20, 0x00};
    int32_t result[20];
    int first_char_witdh[5] = {1, 2, 3, 4, 1};

    for (int i = 0; i < 25; i+=5)
    {
        int32_t ret = 0;
        CHECK(utf8_to_uft32(column_text, i, ret) == first_char_witdh[i/5]);
        CHECK(ret == expected[i/5 * 4]);
    }

    REQUIRE(fill_column(result, column_text, column_details));
    for (int i = 0; i < 20; i++)
    {
        CHECK(result[i] == expected[i]);
    }
}


