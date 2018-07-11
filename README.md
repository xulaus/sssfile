# SSSFile

The Triple S file format is format designed for the sharing of survey data between tools. It was created by the [Triple S Group](http://www.triple-s.org/) and its specification can be found on their website.

This library has been created to read Triple S format files into memory in a highly performant manner, with the lofty goal of being capable of zero copy reading and direct bindings to NumPy and Pandas while maintaining a shared library for other languages.

## Why use Triple S

Triple S adds a metadata layer on top of human readable formats and is a good choice where something like CSV or a custom fixed width format a may have been used otherwise. The standard ensures portablity - which CSVs can struggle with, and the metadata capabilties provide features that are usually only available in binary formats.

### Comparison with other formats

|                      |SSS|CSV|HDF5|SQLite|
|----------------------|:-:|:-:|:--:|:----:|
|Supports Append       | X | X | 1. |  X   |
|Public Standard       | X |2. | X  |  X   |
|Has Metadata          | X |   | X  |  X   |
|Categorical Support   | X |   | X  |      |
|Human Readable        | X | X |    |      |
|Generic Object Support|   |   | X  |      |

1. Depends on configuration
2. While a "standard" exists for CSV, it is often not followed

# Library Details

## Current status

It is still in the early stages of development.

## Why add a custom extension for DAT parsing?

In SSS 3.0 unicode support was added to DAT files, however this was done in a way that requires the parsing of the entire DAT file in order to extract a single column or row. This is because start and end positions are given by character not by byte. As UTF-8 characters are sometimes multiple bytes this requires the parsing of the entire file to find out what character position you are currently at.

The custom DAT layout is on a byte positions. This enables precomputation of each cell positions, and loading just of that cell.

# Development

To build, you will need cmake > 3.2. The build requires RapidXML and the test suite Catch, but the cmake script will download these automatically.

Recomended build is
```
make && make test
```
