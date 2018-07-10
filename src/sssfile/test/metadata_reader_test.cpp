#include "catch.hpp"

#include "sssfile/metadata_reader.h"

using namespace SSSFile;

template<size_t N> size_t static_strlen(const char (&/*unused*/)[N]) { return N; }

const char small_xmldata[] = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                             "<sss version=\"2.0\">\n"
                             "  <date>2017-10-15</date>\n"
                             "  <time>19:57:07</time>\n"
                             "  <origin>An Origin</origin>\n"
                             "  <survey>\n"
                             "    <title>SSS 2.0 Example</title>\n"
                             "    <record ident=\"A\" format=\"fixed\">\n"
                             "      <variable ident=\"00001\" type=\"character\">\n"
                             "        <name>respondent_id</name>\n"
                             "        <label>Respondent</label>\n"
                             "        <size>3</size>\n"
                             "        <position start=\"1\" finish=\"3\"/>\n"
                             "      </variable>\n"
                             "    </record>\n"
                             "  </survey>\n"
                             "</sss>\n\0";

TEST_CASE("Can parse to column_iterator", "[xml]")
{
    column_iterator *iter = nullptr;
    REQUIRE(xml_file_to_column_iterator(small_xmldata, static_strlen(small_xmldata) - 1, &iter) == SUCCESS);
}

TEST_CASE("column_iterator can read column name", "[xml]")
{
    column_iterator *iter = nullptr;
    REQUIRE(xml_file_to_column_iterator(small_xmldata, static_strlen(small_xmldata) - 1, &iter) == SUCCESS);

    char expected_name[] = "respondent_id";

    char buf[255];
    size_t length = find_column_name(iter, buf, 255);
    REQUIRE(length == static_strlen(expected_name) - 1);
    REQUIRE(memcmp(buf, expected_name, length) == 0);

    char buf2[3];
    size_t length2 = find_column_name(iter, buf2, 3);
    REQUIRE(length2 == 3);
    REQUIRE(memcmp(buf2, expected_name, length2) == 0);
}

TEST_CASE("column_iterator can read column ident", "[xml]")
{
    column_iterator *iter = nullptr;
    REQUIRE(xml_file_to_column_iterator(small_xmldata, static_strlen(small_xmldata) - 1, &iter) == SUCCESS);

    char expected_ident[] = "00001";

    char buf[255];
    size_t length = find_column_ident(iter, buf, 255);
    REQUIRE(length == static_strlen(expected_ident) - 1);
    REQUIRE(memcmp(buf, expected_ident, length) == 0);

    char buf2[3];
    size_t length2 = find_column_ident(iter, buf2, 3);
    REQUIRE(length2 == 3);
    REQUIRE(memcmp(buf2, expected_ident, length2) == 0);
}

TEST_CASE("column_iterator can read column label", "[xml]")
{
    column_iterator *iter = nullptr;
    REQUIRE(xml_file_to_column_iterator(small_xmldata, static_strlen(small_xmldata) - 1, &iter) == SUCCESS);

    char expected_label[] = "Respondent";

    char buf[255];
    size_t length = find_column_label(iter, buf, 255);
    REQUIRE(length == static_strlen(expected_label) - 1);
    REQUIRE(memcmp(buf, expected_label, length) == 0);

    char buf2[3];
    size_t length2 = find_column_label(iter, buf2, 3);
    REQUIRE(length2 == 3);
    REQUIRE(memcmp(buf2, expected_label, length2) == 0);
}

TEST_CASE("column_iterator can read column details", "[xml]")
{
    column_iterator *iter = nullptr;
    REQUIRE(xml_file_to_column_iterator(small_xmldata, static_strlen(small_xmldata) - 1, &iter) == SUCCESS);

    sss_column_metadata column_details;
    REQUIRE(find_column_details(iter, &column_details));
    REQUIRE(column_details.type == sss_column_metadata::TYPE_UTF8);
    REQUIRE(column_details.size == 2);
    REQUIRE(column_details.offset == 1);
}
