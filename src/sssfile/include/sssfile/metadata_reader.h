#ifndef __SSSFILE_METADATA_READER_
#define __SSSFILE_METADATA_READER_

#import "sssfile/column_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace SSSFile
{
    bool read_xml_from_substr(const char *const_buffer, const size_t length);
}

#ifdef __cplusplus
}
#endif

#endif
