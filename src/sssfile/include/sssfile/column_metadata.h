#ifndef __SSSFILE_COLUMN_METADATA_
#define __SSSFILE_COLUMN_METADATA_

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
} // namespace SSSFile

#ifdef __cplusplus
}
#endif

#endif
