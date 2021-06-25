import pytest
from pytson.serializer import Serializer
import numpy as np


def test_empty_list():
    try:
        s = Serializer([])
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create empty list object")


def test_empty_map():
    try:
        s = Serializer({})
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create empty map object")


def test_simple_list_with_null():
    try:
        s = Serializer([None])
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_simple_list_with_one():
    try:
        s = Serializer([1])
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_simple_list():
    try:
        s = Serializer(["a", True, False, 42, 42.0])
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_simple_int32_list():
    try:
        s = Serializer(np.array([42, 42], dtype=np.int32))
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_simple_uint32_list():
    try:
        s = Serializer(np.array([42, 42], dtype=np.uint32))
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_simple_cstring_list():
    try:
        s = Serializer(["42", "42.0"])
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_simple_map():
    try:
        s = Serializer({"a": "a", "b": "b"})
        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")


def test_all_types():
    try:
        s = Serializer(
            {
                "null": None,
                "string": "hello",
                "integer": 42,
                "float": 42.0,
                "bool_t": True,
                "bool_f": False,
                "map": {"string": "42"},
                "list": [42, "42", {"string": "42"}, ["42", 42]],
                "uint8": np.array([42, 42], dtype=np.uint8),
                "uint16": np.array([42, 42], dtype=np.uint16),
                "uint32": np.array([42, 42], dtype=np.uint32),
                "int8": np.array([42, 42], dtype=np.int8),
                "uint16": np.array([42, 42], dtype=np.int16),
                "uint32": np.array([42, 42], dtype=np.int32),
                "uint64": np.array([42, 42], dtype=np.int64),
                "float32": np.array([42, 42], dtype=np.float32),
                "float64": np.array([42, 42], dtype=np.float64),
                "cstringlist": ["x", "y", "hello"],
            }
        )

        print(s.readBytes())
        assert True

    except:
        pytest.fail("Cannot create simple list with null")
