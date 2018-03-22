#include <cstdlib>

#include "sssfile/error_codes.h"

namespace SSSFile
{
    const char *SSSError_messages[SSSERROR_COUNT] = {"Successful parse!",
                                                     "Unknown column type.",
                                                     "Column to parse overlaps line end.",
                                                     "Buffer is not cleanly divided by line size.",
                                                     "Could not parse number from buffer.",
                                                     "Could not parse utf8 from buffer. Is it encoded correctly?"
                                                     "Invalid Error Code."};

    const char *get_error_message(SSSError error)
    {
        auto index = static_cast<size_t>(error);
        if (error >= static_cast<size_t>(SSSERROR_COUNT))
        {
            index = static_cast<size_t>(SSSERROR_COUNT);
        }
        return SSSError_messages[index];
    }
} // namespace SSSFile
