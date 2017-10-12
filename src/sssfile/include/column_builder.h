#include <string_view>

namespace SSSFile
{
    struct column_metadata {
        enum {
            TYPE_FLOAT,
            TYPE_INT
        } type;
        size_t line_length;
        size_t size;
        size_t offset;
    };
    std::unique_ptr<int[]> build_integer_column_from_buffer(
        const std::string_view& buffer,
        column_metadata column_details);

    std::unique_ptr<double[]> build_float_column_from_buffer(
        const std::string_view& buffer,
        column_metadata column_details);

    size_t column_length(
        const std::string_view& buffer,
        const column_metadata column_details);
}
