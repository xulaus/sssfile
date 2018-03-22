#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sssfile/metadata_reader.h"

#include "common.h"

#include "rapidxml.hpp"

namespace SSSFile
{

    bool column_from_xml(rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto name_node = variable.first_node("name");
        if(!name_node)
        {
            printf("No name on variable\n");
            return false;
        }

        auto name = name_node->value();
        auto type = variable.first_attribute("type");
        if(!type)
        {
            printf("Variable '%s' has no type declared\n", name);
            return false;
        }

        if(strcmp("character", type->value()) != 0)
        {
            printf("Variable '%s' is of unknown type %s\n", name, type->value());
            return false;
        }

        auto position = variable.first_node("position");
        auto start_node = (position) ? position->first_attribute("start") : nullptr;
        auto end_node = (position) ? position->first_attribute("finish") : nullptr;
        if(!position || !start_node || !end_node)
        {
            printf("Variable '%s' has an invalid position definition\n", name);
            return false;
        }

        unsigned int start, end;
        if(!to_numeric(start_node->value(), start) || !to_numeric(end_node->value(), end))
        {
            printf("Variable '%s' has an invalid position definition\n", name);
            return false;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_UTF8;
        return true;
    }

    bool read_xml_from_substr(const char * const_buffer, const size_t length)
    {
        rapidxml::xml_document<> doc;
        char *buffer = (char *) malloc(sizeof(char) * length);
        if(!buffer) return false;
        // @TODO RapidXML has non modifying version. we should use that
        memcpy(buffer, const_buffer, length);


        doc.parse<0>((char *) buffer);

        auto envelope = doc.first_node("sss");
        auto root_node = envelope ? envelope->first_node("survey") : nullptr;
        auto record = root_node ? root_node->first_node("record") : nullptr;
        if(!record)
        {
            free(buffer);
            return false;
        }

        for(
            auto variable = record->first_node("variable");
            variable;
            variable = variable->next_sibling()
        )
        {
            sss_column_metadata col;
            if(!column_from_xml(*variable, col))
            {
                free(buffer);
                return false;
            }
        }
        free(buffer);
        return true;
    }
}
