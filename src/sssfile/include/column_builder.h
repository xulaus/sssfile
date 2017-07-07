#include <gsl/gsl>

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
    std::unique_ptr<int[]> build_column_from_buffer(
        gsl::span<const char> buffer,
        column_metadata column_details);
}
