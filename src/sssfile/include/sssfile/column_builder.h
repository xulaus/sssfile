#ifndef __SSSFILE_COLUMN_BUILDER_
#define __SSSFILE_COLUMN_BUILDER_

#import "sssfile/column_metadata.h"
#import "sssfile/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace SSSFile
{
    SSSError fill_column_from_substr(void *array, const char *buffer, const size_t length,
                                     const sss_column_metadata &column_details);

    SSSError fill_column_from_cstr(void *array, const char *buffer, const sss_column_metadata &column_details);

    size_t column_length_from_substr(const char *buffer, const size_t length,
                                     const sss_column_metadata &column_details);

    size_t column_length_from_cstr(const char *buffer, const sss_column_metadata &column_details);
} // namespace SSSFile

#ifdef __cplusplus
}
#endif

#endif
