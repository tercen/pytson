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
        if obj is None:
            self.addNull()
        elif isinstance(obj, str):
            self.addString(obj)
        elif isinstance(obj, int):
            self.addInteger(obj)
        elif isinstance(obj, float):
            self.addDouble(obj)
        elif isinstance(obj, bool):
            self.addBool(obj)

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
        print(obj)
        _funcDict = {
            np.dtype("int8"): lambda x: self.addIntList(x, type=spec.LIST_INT8_TYPE),
            np.dtype("uint8"): lambda x: self.addIntList(x, type=spec.LIST_UINT8_TYPE),
            np.dtype("int16"): lambda x: self.addIntList(x, type=spec.LIST_INT16_TYPE),
            np.dtype("uint16"): lambda x: self.addIntList(
                x, type=spec.LIST_UINT16_TYPE
            ),
            np.dtype("int32"): lambda x: self.addIntList(x, type=spec.LIST_INT32_TYPE),
            np.dtype("uint32"): lambda x: self.addIntList(
                x, type=spec.LIST_UINT32_TYPE
            ),
            np.dtype("int64"): lambda x: self.addIntList(x, type=spec.LIST_INT64_TYPE),
            np.dtype("float32"): lambda x: self.addIntList(
                x, type=spec.LIST_FLOAT32_TYPE
            ),
            np.dtype("float64"): lambda x: self.addIntList(
                x, type=spec.LIST_FLOAT64_TYPE
            ),
        }

        f = _funcDict.get(obj.dtype, None)

        if f is None:
            raise ValueError("List type not found.")

        f(obj)

    def addIntList(self, obj, type):
        _l = len(obj)
        self.addType(type)
        self.addLength(_l)
        self.con.write(obj.tobytes())

    def addStringList(self, obj):
        _l = "".join(obj)
        self.addType(spec.LIST_STRING_TYPE)
        self.addLength(len(_l) + 2 * len(obj))
        for x in obj:
            self.addString(x)

    def getBytes(self):
        return self.con
