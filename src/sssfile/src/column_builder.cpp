#include <memory>
#include <cstdio>

#include "column_builder.h"
#include "sss_string.h"

namespace SSSFile
{
    std::unique_ptr<int[]> build_column_from_buffer(
        gsl::span<const char> buffer,
        int col_begin,
        int col_size,
        int line_length)
    {
        assert((col_begin + col_size) < line_length);

        auto buffer_length = buffer.length();
        assert(buffer_length % line_length == 0);

        auto array_size = buffer_length / line_length;

        auto array = std::make_unique<int[]>(array_size);
        for(int i = 0; col_begin < buffer_length; i++, col_begin += line_length)
        {
            auto col = gsl::span<const char>(buffer.data() + col_begin, col_size);
            array[i] = SSSFile::to_i(col);
        }

        return array;
    };
}
