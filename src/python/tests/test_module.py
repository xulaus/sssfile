"""
generic module interaction tests
"""

import os.path
import pytest

import sssfile


def get_abs_path(file_path):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(test_dir, file_path)


def test_unknown_type_error_exists():
    assert issubclass(sssfile.UnknownTypeError, Exception)


def test_failed_to_convert_exception_exists():
    assert issubclass(sssfile.FailedToConvert, Exception)


def test_no_such_file_error_exists():
    assert issubclass(sssfile.NoSuchFileError, Exception)


def test_from_file_when_file_doesnt_exist():
    with pytest.raises(sssfile.NoSuchFileError):
        sssfile.from_file("file/path")


def test_from_file_when_file_does_exist():
    dat_file = get_abs_path("data/sss-2.0.dat")
    sssfile.from_file(dat_file)


def test_from_xmlfile_when_file_doesnt_exist():
    with pytest.raises(sssfile.NoSuchFileError):
        sssfile.from_xmlfile("file/path")


def test_from_xmlfile_when_file_does_exist():
    xml_file = get_abs_path("data/sss-2.0.xml")
    sssfile.from_xmlfile(xml_file)



