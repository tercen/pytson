import os

print(os.getcwd())

from pytson.serializer import Serializer
import numpy as np

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
