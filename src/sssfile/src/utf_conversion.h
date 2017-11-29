#ifndef __SSSFILE_UTF8_CONVERSION_
#define __SSSFILE_UTF8_CONVERSION_

#include <string_view>

namespace SSSFile
{
    int utf8_to_uft32(const std::string_view &buffer, size_t offset, int32_t &out);
}

#endif
