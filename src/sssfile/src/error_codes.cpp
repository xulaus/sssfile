#include <cstdlib>

#include "sssfile/error_codes.h"

namespace SSSFile
{
    const char *get_error_message(SSSError error)
    {
        const char *messages[SSSERROR_COUNT + 1] = {};

        messages[SUCCESS] = "Successful parse!";

        messages[UNKNOWN_TYPE] = "Unknown column type.";
        messages[COLUMN_OVERLAPS_LINE_END] = "Column to parse overlaps line end.";
        messages[BUFFER_WRONG_SIZE] = "Buffer is not cleanly divided by line size.";

        messages[INVALID_NUMBER] = "Could not parse number from buffer.";
        messages[INVALID_UTF8_STRING] = "Could not parse utf8 from buffer. Is it encoded correctly?";

        messages[OOM] = "Not enough memory.";
        messages[END_OF_FILE] = "Reached end of file.";
        messages[INVALID_ARGUMENTS] = "Invalid arguments provided.";

        messages[NO_RECORD_NODE] = "Could not find, record node in given metadata file.";

        messages[VARIABLE_HAS_NO_TYPE_ATTR] = "Variable has no type attribute.";
        messages[VARIABLE_HAS_INVALID_TYPE_ATTR] = "Variable type attribute is invalid.";
        messages[VARIABLE_HAS_TOO_MANY_TYPE_ATTRS] = "Variable has too many type attributes.";
        messages[VARIABLE_HAS_NO_POSITION_NODE] = "Variable has no position node.";
        messages[VARIABLE_HAS_INVALID_POSITION_NODE] = "Variable position node has invalid attributes.";
        messages[VARIABLE_HAS_TOO_MANY_POSITION_NODES] = "Variable has too many position nodes.";

        messages[SSSERROR_COUNT] = "Invalid Error Code.";

        auto index = static_cast<size_t>(error);
        if (error > static_cast<size_t>(SSSERROR_COUNT))
        {
            index = static_cast<size_t>(SSSERROR_COUNT);
        }
        return messages[index];
    }
} // namespace SSSFile
