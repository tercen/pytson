from pytson import encodeTSON, decodeTSON
import numpy as np
from collections.abc import Iterable

s = {
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

tson_bytes = encodeTSON(s)

print(tson_bytes.getvalue())

tson_bytes.seek(0)  # reset stream
tson_obj = decodeTSON(tson_bytes)

for (k1, v1), (k2, v2) in zip(s.items(), tson_obj.items()):
    print("hello")
    print(type(k1), type(k2))
    print(k1 == k2)
    print(v1, isinstance(v1, Iterable))
    try:
        print(all(v1 == v2))
    except:
        print(v1 == v2)
