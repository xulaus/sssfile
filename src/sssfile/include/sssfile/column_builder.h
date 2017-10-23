#ifndef __SSSFILE_COLUMN_BUILDER_
#define __SSSFILE_COLUMN_BUILDER_

#include <string_view>

namespace SSSFile
{
    struct column_metadata {
        enum {
            TYPE_NONE,
            TYPE_DOUBLE,
            TYPE_INT32
        } type;
        size_t line_length;
        size_t size;
        size_t offset;
    };
    std::unique_ptr<int[]> build_integer_column_from_buffer(
        const std::string_view& buffer,
        const column_metadata& column_details);

    std::unique_ptr<double[]> build_float_column_from_buffer(
        const std::string_view& buffer,
        const column_metadata& column_details);

    bool fill_column(
        void * array,
        const std::string_view&  buffer,
        const column_metadata& column_details);

    size_t column_length(
        const std::string_view& buffer,
        const column_metadata& column_details);
}

#endif
