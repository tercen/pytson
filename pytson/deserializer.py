import struct
import numpy as np
import pytson.spec as spec
from pytson.error import TsonError
import sys


# support for py2.x and py3.x+
# most likely we should just drop py2.x at all
from io import BytesIO as StringIO


int_struct = struct.Struct("<i")
double_struct = struct.Struct("<d")
type_struct = struct.Struct("<B")


class DeSerializer:
    def __init__(self, con, mode = "old", chunk=8*1024):
        self.byteChunk = bytes()
        self.chunkPointer = 0
        self.chunkSize = chunk
        self.mode= mode
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
    
    def read(self,  nRead):
        if self.mode == "old":
            return self.con.read(nRead)
        else:
            if (self.chunkPointer + nRead) > self.chunkSize or len(self.byteChunk) < nRead:
                self.read_new_chunk()

         
            i1 = self.chunkPointer
            i2 = self.chunkPointer + nRead 
            
            bts = self.byteChunk[i1:i2]
            
            self.chunkPointer = i2 
            
            return bts
        
    def read_new_chunk(self):
        self.byteChunk = self.byteChunk[self.chunkPointer:] + self.con.read(self.chunkSize)
        self.chunkPointer = 0


    def readType(self):
        return type_struct.unpack(self.read(1))[0]

    def readLength(self):
        _len = int_struct.unpack(self.read(4))[0]
        return _len

    # Add object
    def readObject(self):
        _type = self.readType()

        if _type == spec.BOOL_TYPE:
            obj = self.readBool()
        elif _type == spec.INTEGER_TYPE:
            obj = self.readInteger()
        elif _type == spec.DOUBLE_TYPE:
            obj = self.readDouble()
        elif _type == spec.STRING_TYPE:
            obj = self.readString()[0]
        elif _type == spec.LIST_TYPE:
            obj = self.readList()
        elif _type == spec.MAP_TYPE:
            obj = self.readMap()
        elif _type == spec.LIST_STRING_TYPE:
            obj = self.readStringList()
        elif _type == spec.LIST_UINT8_TYPE:
            obj = self.readTypedNumList(np.uint8, 1)
        elif _type == spec.LIST_UINT16_TYPE:
            obj = self.readTypedNumList(np.uint16, 2)
        elif _type == spec.LIST_UINT32_TYPE:
            obj = self.readTypedNumList(np.uint32, 4)
        elif _type == spec.LIST_INT8_TYPE:
            obj = self.readTypedNumList(np.int8, 1)
        elif _type == spec.LIST_INT16_TYPE:
            obj = self.readTypedNumList(np.int16, 2)
        elif _type == spec.LIST_INT32_TYPE:
            obj = self.readTypedNumList(np.int32, 4)
        elif _type == spec.LIST_INT64_TYPE:
            obj = self.readTypedNumList(np.int64, 8)
        elif _type == spec.LIST_FLOAT32_TYPE:
            obj = self.readTypedNumList(np.float32, 4)
        elif _type == spec.LIST_FLOAT64_TYPE:
            obj = self.readTypedNumList(np.float64, 8)
        elif _type == spec.NULL_TYPE:
            obj = None
        else:
            raise ValueError("List type not found.") # 75

        return obj

    # Basic types (null, string, integer, double, bool)

    def readString(self):
        r = []
        if self.mode == "old":
            bytesRead = 0
            while True:
                b = self.read(1)
                # print(r, b)
                if b == b"\x00":
                    break
                
                
                bytesRead = bytesRead + len(b)
                r.append(b.decode("utf-8", errors="ignore"))

            

            return ["".join(r), bytesRead+1]
        else:
            
            while True:
                b = self.read(1)
                # print(r, b)
                if b == b"\x00":
                    break
               
            
                r.append(b.decode("utf-8", errors="ignore"))

            return ["".join(r), len(r)+1]

        # return "".join([x.decode("utf-8") for x in i])

    def readInteger(self):
        return int_struct.unpack(self.read(4))[0]

    def readDouble(self):
        return double_struct.unpack(self.read(8))[0]

    def readBool(self):
        return type_struct.unpack(self.read(1))[0] > 0

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

    def readTypedNumList(self, type, size):
        l = self.readLength()
        return np.frombuffer(self.read(size * l), dtype=type)

    def readStringList(self):
        l = self.readLength()
        result = []

        # _start = self.con.tell()
        bytesRead = 0

        if l > 0:
            for _ in range(l):
                # if self.con.tell() >= (_start + l):
                if bytesRead >=  l:
                    break

                readRes = self.readString()
                # result.append(self.readString())
                result.append(readRes[0])
                bytesRead = bytesRead + readRes[1]

        return result

    def getObject(self):
        return self.obj