import pytest
from mypkg.math_utils import add, div

def test_add():
    assert add(1,2) == 3

def test_div():
    assert div(6,2) == 3

def test_div_zero():
    with pytest.raises(ZeroDivisionError):
        div(1,0)
