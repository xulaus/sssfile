#include <gsl/gsl>

namespace SSSFile
{
    std::unique_ptr<int[]> build_column_from_buffer(
        gsl::span<const char> buffer,
        int col_begin,
        int col_end,
        int line_length);
}
