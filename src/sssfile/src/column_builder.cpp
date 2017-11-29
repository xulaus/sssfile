#include <cstdio>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <string_view>

#include "sssfile/column_builder.h"

#include "utf_conversion.h"
#include "common.h"

namespace SSSFile
{
    bool validate_column(const std::string_view &buffer, const sss_column_metadata &column_details)
    {
        // Need to consider that the last line may be one byte shorter (no \n)
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto extra = (buffer.length() + no_new_line_at_end) % column_details.line_length;
        return extra == 0;
    }

    template <class Numeric, typename = std::enable_if_t<std::is_arithmetic<Numeric>::value>>
    bool fill_column(Numeric *array, const std::string_view &buffer, const sss_column_metadata &column_details)
    {
        auto col_begin = column_details.offset;
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;
        const auto buffer_length = buffer.length();
        for (int i = 0; col_begin < buffer_length; i++, col_begin += line_length)
        {
            const auto col = std::string_view(buffer.data() + col_begin, col_size);
            if (!to_numeric(col, array[i]))
            {
                break;
            }
        }

        return col_begin >= buffer_length;
    }


    bool copy_from_column(char *array, const std::string_view &buffer, const sss_column_metadata &column_details)
    {
        auto col_begin = column_details.offset;
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;
        const auto buffer_length = buffer.length();
        for (int i = 0; col_begin < buffer_length; i += col_size, col_begin += line_length)
        {
            std::memcpy(array + i, buffer.data() + col_begin, col_size);
        }

        return true;
    }

    bool cast_from_column(int32_t *array, const std::string_view &buffer, const sss_column_metadata &column_details)
    {
        auto col_begin = column_details.offset;
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;
        const auto buffer_length = buffer.length();

        for (int i = 0; col_begin < buffer_length; i += col_size, col_begin += line_length)
        {
            const auto col = std::string_view(buffer.data() + col_begin, col_size);
            int current_code_point = 0;
            for(int j = 0; j < col.length(); current_code_point++)
            {
                int converted = utf8_to_uft32(col, j, array[i + current_code_point]);
                if(converted == 0)
                {
                    return false;
                }
                j += converted;
            }
            while(current_code_point < col_size)
            {
                array[i + current_code_point++] = 0;
            }
        }

        return true;
    }

    size_t column_length(const std::string_view &buffer, const sss_column_metadata &column_details)
    {
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto line_length = column_details.line_length;
        return (buffer.length() + no_new_line_at_end) / line_length;
    }

    size_t column_length(const char *buffer, const size_t length, const sss_column_metadata &column_details)
    {
        return column_length(std::string_view(buffer, length), column_details);
    }


    size_t column_length(const char *buffer, const sss_column_metadata &column_details)
    {
        return column_length(buffer, strlen(buffer), column_details);
    }


    bool fill_column(void *array, const std::string_view &buffer, const sss_column_metadata &column_details)
    {
        if (!validate_column(buffer, column_details))
        {
            return false;
        }

        const auto col_begin = column_details.offset;
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;

        if ((col_begin + col_size) > line_length)
        {
            // Line end must be after the end of the column
            return false;
        }
        switch (column_details.type)
        {
        case sss_column_metadata::TYPE_DOUBLE:
            return fill_column((double *)array, buffer, column_details);
        case sss_column_metadata::TYPE_INT32:
            return fill_column((int32_t *)array, buffer, column_details);
        case sss_column_metadata::TYPE_UTF32:
            return cast_from_column((int32_t *) array, buffer, column_details);
        case sss_column_metadata::TYPE_UTF8:
             return copy_from_column((char *)array, buffer, column_details);
        default:
            return false;
        }
    }

    bool fill_column(void *array, const char *buffer, const size_t length, const sss_column_metadata &column_details)
    {
        return fill_column(array, std::string_view(buffer, length), column_details);
    }

    bool fill_column(void *array, const char *buffer, const sss_column_metadata &column_details)
    {
        return fill_column(array, buffer, strlen(buffer), column_details);
    }
}
