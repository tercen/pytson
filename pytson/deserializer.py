import struct
import numpy as np
import pytson.spec as spec
from pytson.error import TsonError
import sys

# support for py2.x and py3.x+
# most likely we should just drop py2.x at all
try:
    from io import BytesIO as StringIO
except ImportError:
    from cStringIO import StringIO

int_struct = struct.Struct("<I")
double_struct = struct.Struct("<d")
type_struct = struct.Struct("<B")


def _check_list_type(l, _type):
    if len(l) == 0:
        return False

    return all(isinstance(i, _type) for i in l)


class DeSerializer:
    def __init__(self, con):
        if con is None:
            raise TsonError("Connection cannot be None.")

        if sys.getsizeof(con) == 0:
            raise TsonError("Connection buffer is empty.")

        self.con = con

        version = self.readObject()

        if version != spec.TSON_SPEC_VERSION:
            raise TsonError(
                f"TSON version mismatch, found: {version}, expected : {spec.TSON_SPEC_VERSION}"
            )

        self.obj = self.readObject()

    def readType(self):
        return type_struct.unpack(self.con.read(1))[0]

    def readLength(self):
        # to:do - check list length
        _len = int_struct.unpack(self.con.read(4))[0]

        # if _len == 0:
        #     raise TsonError("Found length of zero")

        return _len

    # Add object
    def readObject(self):
        _type = self.readType()
        _typeDict = {
            spec.NULL_TYPE: lambda: None,
            spec.BOOL_TYPE: self.readBool,
            spec.INTEGER_TYPE: self.readInteger,
            spec.DOUBLE_TYPE: self.readDouble,
            spec.STRING_TYPE: self.readString,
            spec.LIST_TYPE: self.readList,
            spec.MAP_TYPE: self.readMap,
            spec.LIST_STRING_TYPE: self.readStringList,
            spec.LIST_UINT8_TYPE: lambda: self.readTypedIntList(np.uint8, 1),
            spec.LIST_UINT16_TYPE: lambda: self.readTypedIntList(np.uint16, 2),
            spec.LIST_UINT32_TYPE: lambda: self.readTypedIntList(np.uint32, 4),
            spec.LIST_INT8_TYPE: lambda: self.readTypedIntList(np.int8, 1),
            spec.LIST_INT16_TYPE: lambda: self.readTypedIntList(np.int16, 2),
            spec.LIST_INT32_TYPE: lambda: self.readTypedIntList(np.int32, 4),
            spec.LIST_INT64_TYPE: lambda: self.readTypedIntList(np.int64, 8),
            spec.LIST_FLOAT32_TYPE: lambda: self.readTypedIntList(np.float32, 4),
            spec.LIST_FLOAT64_TYPE: lambda: self.readTypedIntList(np.float64, 8),
        }

        return _typeDict[_type]()

    # Basic types (null, string, integer, double, bool)

    def readString(self):
        r = []
        while True:
            b = self.con.read(1)
            # print(r, b)
            if b == b"\x00":
                break

            r.append(b.decode("utf-8"))

        return "".join(r)
        # return "".join([x.decode("utf-8") for x in i])

    def readInteger(self):
        return int_struct.unpack(self.con.read(4))[0]

    def readDouble(self):
        return double_struct.unpack(self.con.read(8))[0]

    def readBool(self):
        return type_struct.unpack(self.con.read(1))[0] > 0

    # Basic list
    def readList(self):
        l = self.readLength()
        result = []

        if l > 0:
            result = [self.readObject() for _ in range(l)]

        return result

    # Basic map
    def readMap(self):
        l = self.readLength()
        _d = {}

        if l > 0:
            for i in range(l):
                k = self.readObject()
                if not (isinstance(k, str)):
                    raise TsonError("Key in map is not a string")

                _d[k] = self.readObject()

        return _d

    def readTypedIntList(self, type, size):
        l = self.readLength()
        return np.frombuffer(self.con.read(size * l), dtype=type)

    def readStringList(self):
        l = self.readLength()
        result = []
        _start = self.con.tell()

        if l > 0:
            for _ in range(l):
                if self.con.tell() >= (_start + l):
                    break

                result.append(self.readObject())

        return result

    def getObject(self):
        return self.obj
