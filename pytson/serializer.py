import struct
import numpy as np
import pytson.spec as spec
from pytson.error import TsonError

# support for py2.x and py3.x+
# most likely we should just drop py2.x at all
try:
    from io import BytesIO as StringIO
except ImportError:
    from cStringIO import StringIO


def _check_list_type(l, _type):
    if len(l) == 0:
        return False

    return all(isinstance(i, _type) for i in l)


class Serializer:
    def __init__(self, obj, con=None):
        self.con = con or StringIO()
        self.addString(spec.TSON_SPEC_VERSION)
        self.addObject(obj)

    def addType(self, spec_type):
        self.con.write(struct.pack("<B", spec_type))

    def addLength(self, len):
        # to:do - check list length
        self.con.write(struct.pack("<I", len))

    # Add object
    def addObject(self, obj):
        # Basic types
        # bool must come before int
        if obj is None:
            self.addNull()
        elif isinstance(obj, str):
            self.addString(obj)
        elif isinstance(obj, float):
            self.addDouble(obj)
        elif isinstance(obj, bool):
            self.addBool(obj)
        elif isinstance(obj, int):
            self.addInteger(obj)

        # Int/float lists
        elif isinstance(obj, np.ndarray):
            self.addIntegerList(obj)

        # String and other lists
        elif isinstance(obj, list):
            if _check_list_type(obj, str):
                self.addStringList(obj)
            else:
                self.addList(obj)

        # Maps
        elif isinstance(obj, dict):
            self.addMap(obj)
        else:
            raise TsonError("Unknown object type.")

    # Basic types (null, string, integer, double, bool)
    def addNull(self):
        self.addType(spec.NULL_TYPE)

    def addString(self, obj):
        self.addType(spec.STRING_TYPE)
        self.con.write(struct.pack("{0}s".format(len(obj)), obj.encode("utf-8")))
        self.addNull()

    def addCString(self, obj):
        self.con.write(struct.pack("{0}s".format(len(obj)), obj.encode("utf-8")))
        self.addNull()

    def addInteger(self, obj):
        self.addType(spec.INTEGER_TYPE)
        self.con.write(struct.pack("<i", obj))

    def addDouble(self, obj):
        self.addType(spec.DOUBLE_TYPE)
        self.con.write(struct.pack("<d", obj))

    def addBool(self, obj):
        self.addType(spec.BOOL_TYPE)
        self.con.write(struct.pack("<B", obj))

    # Basic list
    def addList(self, l):
        self.addType(spec.LIST_TYPE)
        self.addLength(len(l))

        for o in l:  # loop through objects
            self.addObject(o)

    # Basic map
    def addMap(self, m):
        self.addType(spec.MAP_TYPE)
        self.addLength(len(m))

        for k, v in m.items():
            if not (isinstance(k, str)):
                raise TsonError("Map key must be a String.")

            self.addObject(k)
            self.addObject(v)

    # Integer lists
    def addIntegerList(self, obj):
        if obj.dtype == np.dtype("int8"):
            self.addTypedNumList(obj, type=spec.LIST_INT8_TYPE)
        elif obj.dtype == np.dtype("uint8"):
            self.addTypedNumList(obj, type=spec.LIST_UINT8_TYPE)
        elif obj.dtype == np.dtype("int16"):
            self.addTypedNumList(obj, type=spec.LIST_INT16_TYPE)
        elif obj.dtype == np.dtype("uint16"):
            self.addTypedNumList(obj, type=spec.LIST_UINT16_TYPE)
        elif obj.dtype == np.dtype("int32"):
            self.addTypedNumList(obj, type=spec.LIST_INT32_TYPE)
        elif obj.dtype == np.dtype("uint32"):
            self.addTypedNumList(obj, type=spec.LIST_UINT32_TYPE)
        elif obj.dtype == np.dtype("uint64"):
            self.addTypedNumList(obj, type=spec.LIST_UINT64_TYPE)
        elif obj.dtype == np.dtype("int64"):
            self.addTypedNumList(obj, type=spec.LIST_INT64_TYPE)
        elif obj.dtype == np.dtype("float32"):
            self.addTypedNumList(obj, type=spec.LIST_FLOAT32_TYPE)
        elif obj.dtype == np.dtype("float64"):
            self.addTypedNumList(obj, type=spec.LIST_FLOAT64_TYPE)
        else:
            raise ValueError("List type not found.")

    def addTypedNumList(self, obj, type):
        _l = len(obj)
        self.addType(type)
        self.addLength(_l)
        self.con.write(obj.tobytes())

    def addStringList(self, obj):
        count_bytes = 0
        for my_str in obj:
            count_bytes = count_bytes + len(my_str.encode("utf-8"))
        self.addType(spec.LIST_STRING_TYPE)
        self.addLength(count_bytes + len(obj))
        for x in obj:
            self.addCString(x)

    def getBytes(self):
        return self.con
