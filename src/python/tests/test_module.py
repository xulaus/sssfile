"""
generic module interaction tests
"""

import pytest
import sssfile

def test_from_file_when_file_doesnt_exist():
    with pytest.raises(Exception):
        sssfile.from_file("file/path")

