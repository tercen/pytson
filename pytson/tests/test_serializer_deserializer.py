import unittest
from pytson import encodeTSON, decodeTSON
import numpy as np


class TestSerDeserTson(unittest.TestCase):
    def test_empty_map(self):
        obj = {}
        tson_bytes = encodeTSON(obj)
        tson_bytes.seek(0)  # reset stream
        tson_obj = decodeTSON(tson_bytes)

        self.assertDictEqual(tson_obj, obj)


    def test_serializer_deserializer_object(self):
        obj = {
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

        tson_bytes = encodeTSON(obj)

        tson_bytes.seek(0)  # reset stream
        tson_obj = decodeTSON(tson_bytes)

        # self.assertDictEqual(tson_obj, obj)

        res = []

        for (k1, v1), (k2, v2) in zip(obj.items(), tson_obj.items()):
            res.append(k1 == k2)
            try:
                res.append(all(v1 == v2))
            except:
                res.append(v1 == v2)

        assert all(res)
