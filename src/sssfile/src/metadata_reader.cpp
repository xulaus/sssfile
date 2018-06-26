#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sssfile/metadata_reader.h"

#include "common.h"

#include "rapidxml.hpp"

namespace SSSFile
{
    bool unpack_position_node(const char *variable_name, rapidxml::xml_node<> *position, unsigned int &start,
                              unsigned int &end)
    {
        auto start_node = (position) != nullptr ? position->first_attribute("start") : nullptr;
        auto end_node = (position) != nullptr ? position->first_attribute("finish") : nullptr;
        if ((start_node == nullptr) || (end_node == nullptr))
        {
            printf("Variable '%s' has an invalid position definition\n", variable_name);
            return false;
        }

        if (!to_numeric(start_node->value(), start) || !to_numeric(end_node->value(), end))
        {
            printf("Variable '%s' has an invalid position definition\n", variable_name);
            return false;
        }
        return true;
    }

    bool parse_quantity_column(const char *variable_name, rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto position = variable.first_node("position");
        unsigned int start, end;
        if (!unpack_position_node(variable_name, position, start, end))
        {
            return false;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_DOUBLE;
        return true;
    }

    bool parse_character_column(const char *variable_name, rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto position = variable.first_node("position");

        unsigned int start, end;
        if (!unpack_position_node(variable_name, position, start, end))
        {
            return false;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_UTF8;
        return true;
    }

    bool column_from_xml(rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto name_node = variable.first_node("name");
        if (name_node == nullptr)
        {
            printf("No name on variable\n");
            return false;
        }

        auto name = name_node->value();
        auto type = variable.first_attribute("type");
        if (type == nullptr)
        {
            printf("Variable '%s' has no type declared\n", name);
            return false;
        }

        if (strcmp("character", type->value()) == 0)
        {
            return parse_character_column(name, variable, column);
        }
        else if (strcmp("quantity", type->value()) == 0)
        {
            return parse_quantity_column(name, variable, column);
        }
        else
        {
            printf("Variable '%s' is of unknown type %s\n", name, type->value());
            return false;
        }
    }

    bool read_xml_from_substr(const char *const_buffer, const size_t length)
    {
        auto *buffer = static_cast<char *>(malloc(sizeof(char) * (length + 1)));
        rapidxml::xml_document<> doc;
        if (buffer == nullptr)
        {
            return false;
        }
        // @TODO RapidXML has non modifying version. we should use that
        memcpy(buffer, const_buffer, length);
        buffer[length] = '\0';

        doc.parse<0>(buffer);

        auto envelope = doc.first_node("sss");
        auto root_node = envelope != nullptr ? envelope->first_node("survey") : nullptr;
        auto record = root_node != nullptr ? root_node->first_node("record") : nullptr;
        if (record == nullptr)
        {
            free(buffer);
            return false;
        }

        for (auto variable = record->first_node("variable"); variable != nullptr; variable = variable->next_sibling())
        {
            sss_column_metadata col{};
            if (!column_from_xml(*variable, col))
            {
                free(buffer);
                return false;
            }
        }
        free(buffer);
        return true;
    }
} // namespace SSSFile
