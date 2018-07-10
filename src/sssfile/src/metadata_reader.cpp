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

    bool parse_single_column(const char *variable_name, rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto position = variable.first_node("position");

        unsigned int start, end;
        if (!unpack_position_node(variable_name, position, start, end))
        {
            return false;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_INT32;
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
        else if (strcmp("single", type->value()) == 0)
        {
            return parse_quantity_column(name, variable, column);
        }
        else if (strcmp("multiple", type->value()) == 0)
        {
            printf("Variable '%s' is multiple select. These are not currently supported\n", name);
            return true;
        }
        else
        {
            printf("Variable '%s' is of unknown type %s\n", name, type->value());
            return false;
        }
    }

    struct column_iterator
    {
        rapidxml::xml_document<> document;
        rapidxml::xml_node<> *cur_variable;
        char *xml_string_buffer;
    };

    void free_column_iterator(column_iterator *iter)
    {
        if (iter)
        {
            if (iter->xml_string_buffer)
                free(iter->xml_string_buffer);
            free(iter);
        }
    }

    SSSError xml_file_to_column_iterator(const char *const_buffer, const size_t length, column_iterator **iter_ptr)
    {
        if (!iter_ptr)
        {
            return INVALID_ARGUMENTS;
        }

        column_iterator *iter = static_cast<column_iterator *>(malloc(sizeof(column_iterator)));
        if (!iter)
        {
            return OOM;
        }

        iter->cur_variable = nullptr;
        iter->xml_string_buffer = static_cast<char *>(malloc(sizeof(char) * (length + 1)));
        if (iter->xml_string_buffer == nullptr)
        {
            free_column_iterator(iter);
            return OOM;
        }

        // @TODO RapidXML has non modifying version. we should use that
        memcpy(iter->xml_string_buffer, const_buffer, length);
        iter->xml_string_buffer[length] = '\0';
        iter->document.parse<0>(iter->xml_string_buffer);

        auto envelope = iter->document.first_node("sss");
        auto root_node = envelope != nullptr ? envelope->first_node("survey") : nullptr;
        auto record = root_node != nullptr ? root_node->first_node("record") : nullptr;

        if (record == nullptr)
        {
            free_column_iterator(iter);
            return NO_RECORD_NODE;
        }

        iter->cur_variable = record->first_node("variable");
        *iter_ptr = iter;
        return SUCCESS;
    }

    int find_column_label(column_iterator *iter, char *buffer, size_t length)
    {
        auto label_node = iter->cur_variable->first_node("label");
        if (label_node)
        {
            auto end_ptr = strncpy(buffer, label_node->value(), length);
            return end_ptr == buffer ? strlen(buffer) : end_ptr - buffer;
        }
        else
        {
            return -1;
        }
    }

    int find_column_name(column_iterator *iter, char *buffer, size_t length)
    {
        auto name_node = iter->cur_variable->first_node("name");
        if (name_node)
        {
            auto end_ptr = strncpy(buffer, name_node->value(), length);
            return end_ptr == buffer ? strlen(buffer) : end_ptr - buffer;
        }
        else
        {
            return -1;
        }
    }

    int find_column_ident(column_iterator *iter, char *buffer, size_t length)
    {
        auto ident_attr = iter->cur_variable->first_attribute("ident");
        if (ident_attr)
        {
            auto end_ptr = strncpy(buffer, ident_attr->value(), length);
            return end_ptr == buffer ? strlen(buffer) : end_ptr - buffer;
        }
        else
        {
            return -1;
        }
    }

    bool find_column_details(column_iterator *iter, sss_column_metadata *col)
    {
        bool bad_args = !iter || !col;
        return !bad_args && column_from_xml(*iter->cur_variable, *col);
    }

    bool next_column(column_iterator *iter)
    {
        if (!iter || !iter->cur_variable)
            return false;
        iter->cur_variable = iter->cur_variable->next_sibling();
        return iter->cur_variable != nullptr;
    }
} // namespace SSSFile
