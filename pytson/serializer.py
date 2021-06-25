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
        print("Writing list")
        print(self.con.getvalue())
        self.addType(spec.LIST_TYPE)
        print(self.con.getvalue())
        self.addLength(len(l))
        print("With length")
        print(self.con.getvalue())

        for o in l:  # loop through objects
            print(f"Object writing now: {o}")
            self.addObject(o)
            print(self.con.getvalue())

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
            np.dtype("int8"): self.addInt8List,
            np.dtype("uint8"): self.addUInt8List,
            np.dtype("int16"): self.addInt16List,
            np.dtype("uint16"): self.addUInt16List,
            np.dtype("int32"): self.addInt32List,
            np.dtype("uint32"): self.addUInt32List,
            np.dtype("int64"): self.addInt64List,
            np.dtype("float32"): self.addFloat32List,
            np.dtype("float64"): self.addFloat64List,
        }

        f = _funcDict.get(obj.dtype, None)

        if f is None:
            raise ValueError("List type not found.")

        f(obj)

    def addInt8List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_INT8_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}b".format(_l), *tuple(obj)))

    def addUInt8List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_UINT8_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}B".format(_l), *tuple(obj)))

    def addInt16List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_INT16_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}h".format(_l), *tuple(obj)))

    def addUInt16List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_UINT16_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}H".format(_l), *tuple(obj)))

    def addInt32List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_INT32_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}i".format(_l), *tuple(obj)))

    def addUInt32List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_UINT32_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}I".format(_l), *tuple(obj)))

    def addInt64List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_INT64_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}q".format(_l), *tuple(obj)))

    def addFloat32List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_FLOAT32_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}f".format(_l), *tuple(obj)))

    def addFloat64List(self, obj):
        _l = len(obj)
        self.addType(spec.LIST_FLOAT64_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("<{0}d".format(_l), *tuple(obj)))

    def addStringList(self, obj):
        _l = len(obj)
        obj = "".join(obj)
        self.addType(spec.LIST_STRING_TYPE)
        self.addLength(_l)
        self.con.write(struct.pack("{0}s".format(len(obj)), obj.encode("utf-8")))

    def readBytes(self, seek=True):
        if seek:
            self.con.seek(0)

        return self.con.read()
