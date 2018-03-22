#include "catch.hpp"

#include "sssfile/metadata_reader.h"

using namespace SSSFile;

template<size_t N> size_t static_strlen(const char (&)[N]) { return N; }
TEST_CASE("XML READ", "[xml]")
{
    const char xmldata[] = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
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
    REQUIRE(read_xml_from_substr(xmldata, static_strlen(xmldata) - 1));
}
