import os
import pytest
import tempfile
import shutil
import numpy as np
from utils.data import is_safe_filename

def test_is_safe_filename():
    assert is_safe_filename("test.mp4") is True
    assert is_safe_filename("test.png") is True
    assert is_safe_filename(".hidden.txt") is False
    assert is_safe_filename("../test.txt") is False
    assert is_safe_filename("./test.txt") is False
    assert is_safe_filename("") is False
    assert is_safe_filename(None) is False
