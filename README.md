# SSSFile

The Triple S file format is format designed for the sharing of survey data between tools. It was created by the [Triple S Group](http://www.triple-s.org/) and its specification can be found on their website.

This library has been created to read Triple S format files into memory in a highly performant manner, with the lofty goal of being capable of zero copy reading and direct bindings to NumPy and Pandas while maintaining a shared library for other languages.

## Current status

It is still in the early stages of development.

## Why the custom extension to DAT parsing

In SSS 3.0 unicode support was added to DAT files, however this was done in a way that requires the parsing of the entire DAT file in order to extract a single column or row. This is because start and end positions are given by character not by byte. As UTF-8 characters are sometimes multiple bytes this requires the parsing of the entire file to find out what character position you are currently at.

The custom DAT layout is on a byte positions. This enables precomputation of each cell positions, and loading just of that cell.

# Development

To build, you will need cmake > 3.2. The build requires RapidXML and the test suite Catch, but the cmake script will download these automatically.

Recomended build is
```
make && make test
```
