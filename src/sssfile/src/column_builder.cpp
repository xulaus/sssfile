#include <memory>
#include <cstdio>

#include "column_builder.h"
#include "sss_string.h"

namespace SSSFile
{
    size_t column_length(
        gsl::span<const char> buffer,
        const column_metadata column_details)
    {
        const auto line_length = column_details.line_length;
        assert((buffer.length() % line_length == 0 ) && "Buffer must only contain complete rows");

        return buffer.length() / line_length;
    }

    void fill_column(
        int * array,
        gsl::span<const char> buffer,
        const column_metadata column_details)
    {
        auto col_begin = column_details.offset;
        const auto buffer_length = buffer.length();
        const auto line_length = column_details.line_length;
        const auto col_size = column_details.size;

        assert(((col_begin + col_size) < line_length) && "Line end must be after the end of the column");
        assert((buffer_length % line_length == 0) && "Buffer must only contain complete rows");
        for(int i = 0; col_begin < buffer_length; i++, col_begin += column_details.line_length)
        {
            auto col = gsl::span<const char>(buffer.data() + col_begin, col_size);
            array[i] = SSSFile::to_i(col);
        }
    }

    std::unique_ptr<int[]> build_column_from_buffer(
        gsl::span<const char> buffer,
        const column_metadata column_details)
    {
        auto array = std::make_unique<int[]>(column_length(buffer, column_details));
        fill_column(array.get(), buffer, column_details);
        return array;
    };
}
