#ifndef __SSSFILE_COLUMN_BUILDER_
#define __SSSFILE_COLUMN_BUILDER_

#ifdef __cplusplus
extern "C" {
#endif

namespace SSSFile
{
    struct sss_column_metadata
    {
        enum
        {
            TYPE_NONE,
            TYPE_DOUBLE,
            TYPE_INT32,
            TYPE_UTF8,
            TYPE_UTF32
        } type;
        size_t line_length;
        size_t size;
        size_t offset;
    };

    bool fill_column_from_substr(void *array, const char *buffer, const size_t length, const sss_column_metadata &column_details);

    bool fill_column_from_cstr(void *array, const char *buffer, const sss_column_metadata &column_details);

    size_t column_length_from_substr(const char *buffer, const size_t length, const sss_column_metadata &column_details);

    size_t column_length_from_cstr(const char *buffer, const sss_column_metadata &column_details);

}

#ifdef __cplusplus
}
#endif

#endif
