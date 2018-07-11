#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sssfile/metadata_reader.h"

#include "common.h"

#include "rapidxml.hpp"

namespace SSSFile
{
    SSSError unpack_position_node(rapidxml::xml_node<> *position, unsigned int &start, unsigned int &end)
    {
        if (position->next_sibling("position"))
        {
            return VARIABLE_HAS_TOO_MANY_POSITION_NODES;
        }
        auto start_node = (position) != nullptr ? position->first_attribute("start") : nullptr;
        auto end_node = (position) != nullptr ? position->first_attribute("finish") : nullptr;
        if ((start_node == nullptr) || (end_node == nullptr))
        {
            return VARIABLE_HAS_NO_POSITION_NODE;
        }

        if (!to_numeric(start_node->value(), start) || !to_numeric(end_node->value(), end))
        {
            return VARIABLE_HAS_INVALID_POSITION_NODE;
        }
        return SUCCESS;
    }

    SSSError parse_quantity_column(rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto position = variable.first_node("position");
        unsigned int start, end;
        auto position_unpack = unpack_position_node(position, start, end);
        if (position_unpack != SUCCESS)
        {
            return position_unpack;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_DOUBLE;
        return SUCCESS;
    }

    SSSError parse_character_column(rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto position = variable.first_node("position");

        unsigned int start, end;
        auto position_unpack = unpack_position_node(position, start, end);
        if (position_unpack != SUCCESS)
        {
            return position_unpack;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_UTF8;
        return SUCCESS;
    }

    SSSError parse_single_column(rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto position = variable.first_node("position");

        unsigned int start, end;
        auto position_unpack = unpack_position_node(position, start, end);
        if (position_unpack != SUCCESS)
        {
            return position_unpack;
        }

        column.offset = start;
        column.size = end - start;
        column.type = sss_column_metadata::TYPE_INT32;
        return SUCCESS;
    }

    SSSError column_from_xml(rapidxml::xml_node<> &variable, sss_column_metadata &column)
    {
        auto type = variable.first_attribute("type");
        if (type == nullptr)
        {
            return VARIABLE_HAS_NO_TYPE_ATTR;
        }

        if (type->next_attribute("type"))
        {
            return VARIABLE_HAS_TOO_MANY_TYPE_ATTRS;
        }

        if (strcmp("character", type->value()) == 0)
        {
            return parse_character_column(variable, column);
        }
        else if (strcmp("quantity", type->value()) == 0)
        {
            return parse_quantity_column(variable, column);
        }
        else if (strcmp("single", type->value()) == 0)
        {
            return parse_quantity_column(variable, column);
        }
        else if (strcmp("multiple", type->value()) == 0)
        {
            return UNKNOWN_TYPE;
        }
        else
        {
            return UNKNOWN_TYPE;
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

    SSSError find_column_details(column_iterator *iter, sss_column_metadata *col)
    {
        if (!iter || !col)
        {
            return INVALID_ARGUMENTS;
        }
        return column_from_xml(*iter->cur_variable, *col);
    }

    SSSError next_column(column_iterator *iter)
    {
        if (!iter)
        {
            return INVALID_ARGUMENTS;
        }

        if (!iter->cur_variable)
        {
            return END_OF_FILE;
        }

        iter->cur_variable = iter->cur_variable->next_sibling("variable");
        return iter->cur_variable != nullptr ? SUCCESS : END_OF_FILE;
    }
} // namespace SSSFile
