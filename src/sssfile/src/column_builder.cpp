#include <memory>
#include <cstdio>
#include <type_traits>

#include "sssfile/column_builder.h"
#include "sssfile/sss_string.h"

#include "common.h"

namespace SSSFile
{
    void validate_column(
        const std::string_view& buffer,
        const column_metadata& column_details)
    {
        // Need to consider that the last line may be one byte shorter (no \n)
        const auto no_new_line_at_end = (buffer[buffer.length() - 1] != '\n') ? 1 : 0;
        const auto extra = (buffer.length() + no_new_line_at_end) % column_details.line_length;
        assert((extra == 0) && "Buffer must only contain complete rows");
    }

    size_t column_length(
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        const auto line_length = column_details.line_length;
        validate_column(buffer, column_details);
        return buffer.length() / line_length;
    }

    template <
        class T,
        typename = std::enable_if_t<std::is_signed<T>::value>
    >
    T to_number(std::string_view  buffer) {
        if(std::is_floating_point<T>::value)
        {
            return SSSFile::to_f(buffer);
        }
        else
        {
            return SSSFile::to_i(buffer);
        }
    }

    template <class Numeric, typename = std::enable_if_t<std::is_arithmetic<Numeric>::value>>
    void fill_column(
        Numeric * array,
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        auto col_begin = column_details.offset;
        const auto buffer_length = buffer.length();
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;

        assert(((col_begin + col_size) < line_length) && "Line end must be after the end of the column");
        validate_column(buffer, column_details);

        for(int i = 0; col_begin < buffer_length; i++, col_begin += column_details.line_length)
        {
            const auto& col = std::string_view(buffer.data() + col_begin, col_size);
            array[i] = to_number<Numeric>(col);
        }

    }

    std::unique_ptr<double[]> build_float_column_from_buffer(
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        auto array = std::make_unique<double[]>(column_length(buffer, column_details));

        fill_column<double>(array.get(), buffer, column_details);
        return array;
    };

    std::unique_ptr<int[]> build_integer_column_from_buffer(
        const std::string_view&  buffer,
        const column_metadata& column_details)
    {
        auto array = std::make_unique<int[]>(column_length(buffer, column_details));
        fill_column(array.get(), buffer, column_details);
        return array;
    };
}
