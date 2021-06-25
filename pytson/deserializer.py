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
        return struct.unpack("<B", self.con.read(1))[0]

    def readLength(self):
        # to:do - check list length
        _len = struct.unpack("<I", self.con.read(4))[0]

        if _len == 0:
            raise TsonError("Found length of zero")

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
            spec.LIST_UINT8_TYPE: lambda: self.readTypedIntList("<{0}B", 1),
            spec.LIST_UINT16_TYPE: lambda: self.readTypedIntList("<{0}H", 2),
            spec.LIST_UINT32_TYPE: lambda: self.readTypedIntList("<{0}I", 4),
            spec.LIST_UINT8_TYPE: lambda: self.readTypedIntList("<{0}b", 1),
            spec.LIST_UINT16_TYPE: lambda: self.readTypedIntList("<{0}h", 2),
            spec.LIST_UINT32_TYPE: lambda: self.readTypedIntList("<{0}i", 4),
            spec.LIST_INT64_TYPE: lambda: self.readTypedIntList("<{0}q", 8),
            spec.LIST_FLOAT32_TYPE: lambda: self.readTypedIntList("<{0}f", 4),
            spec.LIST_FLOAT64_TYPE: lambda: self.readTypedIntList("<{0}d", 8),
        }

        # print(f"Found type {_type}")
        # print(f"Associated function {_typeDict[_type]}")
        return _typeDict[_type]()

    # Basic types (null, string, integer, double, bool)

    def readString(self):
        i = iter(lambda: self.con.read(1), b"\x00")
        return "".join([x.decode("utf-8") for x in i])

    def readInteger(self):
        return struct.unpack("<I", self.con.read(4))[0]

    def readDouble(self):
        return struct.unpack("<d", self.con.read(8))[0]

    def readBool(self):
        return struct.unpack("<B", self.con.read(1))[0]

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

    def readTypedIntList(self, fmt, size):
        print(fmt)
        print(size)
        l = self.readLength()
        print(l)

        return struct.unpack(fmt.format(l), self.con.read(size * l))

    def readStringList(self):
        l = self.readLength()
        print(f"Reading string list with length {l}")

        if l > 0:
            return [self.readObject() for _ in range(l)]
        else:
            return []
