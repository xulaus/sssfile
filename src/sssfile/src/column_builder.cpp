#include <memory>
#include <cstdio>
#include <type_traits>
#include <iostream>

#include "sssfile/column_builder.h"

#include "common.h"

namespace SSSFile
{
    bool validate_column(
        const std::string_view& buffer,
        const column_metadata& column_details)
    {
        // Need to consider that the last line may be one byte shorter (no \n)
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto extra = (buffer.length() + no_new_line_at_end) % column_details.line_length;
        return extra == 0;
    }

    size_t column_length(
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto line_length = column_details.line_length;
        return (buffer.length() + no_new_line_at_end) / line_length;
    }

    template <class Numeric, typename = std::enable_if_t<std::is_arithmetic<Numeric>::value>>
    bool fill_column(
        Numeric * array,
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        if(!validate_column(buffer, column_details))
        {
            return false;
        }

        auto col_begin = column_details.offset;
        const auto buffer_length = buffer.length();
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;

        if((col_begin + col_size) > line_length)
        {
            // Line end must be after the end of the column
            return false;
        }

        bool successful = true;
        for(int i = 0; successful && col_begin < buffer_length; i++, col_begin += line_length)
        {
            const auto& col = std::string_view(buffer.data() + col_begin, col_size);
            successful = to_numeric(col, array[i]);
        }

        return successful;
    }

    bool fill_column(
        void * array,
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        switch(column_details.type)
        {
            case column_metadata::TYPE_DOUBLE:
                return fill_column((double *) array, buffer, column_details);
            case column_metadata::TYPE_INT32:
                return fill_column((int32_t *) array, buffer, column_details);
        }
        return false;
    }

    std::unique_ptr<double[]> build_float_column_from_buffer(
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        auto array = std::make_unique<double[]>(column_length(buffer, column_details));

        if(fill_column<double>(array.get(), buffer, column_details))
        {
            return array;
        }
        else
        {
            return nullptr;
        }
    };

    std::unique_ptr<int[]> build_integer_column_from_buffer(
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        auto array = std::make_unique<int[]>(column_length(buffer, column_details));
        if(fill_column(array.get(), buffer, column_details))
        {
            return array;
        }
        else
        {
            return nullptr;
        }
    };
}
