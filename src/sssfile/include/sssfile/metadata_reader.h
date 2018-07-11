#ifndef __SSSFILE_METADATA_READER_
#define __SSSFILE_METADATA_READER_

#import "sssfile/column_metadata.h"
#import "sssfile/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace SSSFile
{
    struct column_iterator;
    SSSError xml_file_to_column_iterator(const char *const_buffer, const size_t length, column_iterator **iter);
    SSSError next_column(column_iterator *iter);
    void free_column_iterator(column_iterator *iter);

    int find_column_label(column_iterator *iter, char *buffer, size_t length);
    int find_column_name(column_iterator *iter, char *buffer, size_t length);
    int find_column_ident(column_iterator *iter, char *buffer, size_t length);

    SSSError find_column_details(column_iterator *iter, sss_column_metadata *column);
}

#ifdef __cplusplus
}
#endif

#endif
