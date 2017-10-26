#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>

#include "sssfile/column_builder.h"

#include "common.h"

namespace SSSFile
{
    bool validate_column(const std::string_view &buffer, const column_metadata &column_details)
    {
        // Need to consider that the last line may be one byte shorter (no \n)
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto extra = (buffer.length() + no_new_line_at_end) % column_details.line_length;
        return extra == 0;
    }

    size_t column_length(const std::string_view &buffer, const column_metadata &column_details)
    {
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto line_length = column_details.line_length;
        return (buffer.length() + no_new_line_at_end) / line_length;
    }

    template <class Numeric, typename = std::enable_if_t<std::is_arithmetic<Numeric>::value>>
    bool fill_column(Numeric *array, const std::string_view &buffer, const column_metadata &column_details)
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


    bool copy_from_column(int32_t *array, const std::string_view &buffer, const column_metadata &column_details)
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

    const int UTF8_seq_length[256] = {
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,5,5,5,5,6,6,1,1
    };


    int utf8_to_uft32(const std::string_view &buffer, size_t offset, int32_t &out)
    {
        int i = offset;
        unsigned char c = reinterpret_cast<const unsigned char&>(buffer[i]);
        int seq_length = UTF8_seq_length[c];

        if(buffer.length() < (i + seq_length))
        {
            return 0;
        }

        int32_t ret = buffer[i++] & ((1 << (8 - seq_length)) - 1);
        while(--seq_length)
        {
            if((buffer[i] & 0xC0) != 0x80)
            {
                // overshoot or invalid UTF-8 tail
                return 0;
            }
            ret = (ret << 6) | (buffer[i++] & 0x3F);
        }
        out = ret;
        return i - offset;
    }

    bool cast_from_column(int32_t *array, const std::string_view &buffer, const column_metadata &column_details)
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

    bool fill_column(void *array, const std::string_view &buffer, const column_metadata &column_details)
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
        case column_metadata::TYPE_DOUBLE:
            return fill_column((double *)array, buffer, column_details);
        case column_metadata::TYPE_INT32:
            return fill_column((int32_t *)array, buffer, column_details);
        case column_metadata::TYPE_UTF32:
            return cast_from_column((int32_t *) array, buffer, column_details);
        case column_metadata::TYPE_UTF8:
             return copy_from_column((int32_t *)array, buffer, column_details);
        default:
            return false;
        }
    }

    std::unique_ptr<double[]> build_float_column_from_buffer(const std::string_view &buffer,
                                                             const column_metadata &column_details)
    {
        auto array = std::make_unique<double[]>(column_length(buffer, column_details));

        if (fill_column<double>(array.get(), buffer, column_details))
        {
            return array;
        }
        else
        {
            return nullptr;
        }
    };

    std::unique_ptr<int[]> build_integer_column_from_buffer(const std::string_view &buffer,
                                                            const column_metadata &column_details)
    {
        auto array = std::make_unique<int[]>(column_length(buffer, column_details));
        if (fill_column(array.get(), buffer, column_details))
        {
            return array;
        }
        else
        {
            return nullptr;
        }
    };
}
