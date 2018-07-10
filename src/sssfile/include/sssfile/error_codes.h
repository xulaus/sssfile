#ifndef __SSSFILE_ERROR_CODES_
#define __SSSFILE_ERROR_CODES_

#ifdef __cplusplus
extern "C" {
#endif

namespace SSSFile
{
    enum SSSError
    {
        SUCCESS = 0,
        UNKNOWN_TYPE,
        COLUMN_OVERLAPS_LINE_END,
        BUFFER_WRONG_SIZE,
        INVALID_NUMBER,
        INVALID_UTF8_STRING,
        OOM,
        INVALID_ARGUMENTS,
        NO_RECORD_NODE,
        SSSERROR_COUNT
    };

    const char *get_error_message(SSSError error);
} // namespace SSSFile

#ifdef __cplusplus
}
#endif

#endif
