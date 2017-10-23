"""
generic module interaction tests
"""

import pytest
import sssfile


def test_from_file_when_file_doesnt_exist():
    with pytest.raises(Exception):
        sssfile.from_file("file/path")


def test_from_file_when_file_does_exist():
    sssfile.from_file("tests/data/sss-2.0.dat")


def test_no_such_file_error_exists():
    assert issubclass(sssfile.NoSuchFileError, Exception)

