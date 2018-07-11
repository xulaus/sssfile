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
        END_OF_FILE,
        INVALID_ARGUMENTS,

        NO_RECORD_NODE,

        VARIABLE_HAS_NO_TYPE_ATTR,
        VARIABLE_HAS_INVALID_TYPE_ATTR,
        VARIABLE_HAS_TOO_MANY_TYPE_ATTRS,
        VARIABLE_HAS_NO_POSITION_NODE,
        VARIABLE_HAS_INVALID_POSITION_NODE,
        VARIABLE_HAS_TOO_MANY_POSITION_NODES,

        SSSERROR_COUNT
    };

    const char *get_error_message(SSSError error);
} // namespace SSSFile

#ifdef __cplusplus
}
#endif

#endif
