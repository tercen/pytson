import struct, math
import numpy as np
import pytson.spec as spec
from pytson.error import TsonError
# from line_profiler import profile
from itertools import compress

# support for py2.x and py3.x+
# most likely we should just drop py2.x at all
import io
from io import BytesIO as StringIO

STRING_LIST=0
NUMERIC_LIST=1
MIXED_LIST=2
MIXED_NUMERIC_LIST=3




class SerializerIt:
    def addTsonSpec(self):
        self.addString(spec.TSON_SPEC_VERSION)

    def __init__(self, con=io.BytesIO()):
        self.con = con
        self.numListType = None

    def addType(self, spec_type):
        self.con.write(struct.pack("<B", spec_type))

    def addLength(self, length):
        # to:do - check list length
        self.con.write(struct.pack("<I", length))



    def __listdtype(self, obj, typeList):
        # Assumptions
        #  1. List has been checked to ensure all ae of the same type
        #  2. Values will only match one type within typeList
        o = obj[0]

        typeCheck = [isinstance(o, t) for t in typeList ]

        res = [i for i, val in enumerate(typeCheck) if val]
        return typeList[res[0]]


            

    # Basic types (null, string, integer, double, bool)
    def addNull(self):
        self.addType(spec.NULL_TYPE)
        return self.con.tell()

    def addString(self, obj):
        self.addType(spec.STRING_TYPE)
        
        self.con.write(struct.pack("{0}s".format(len(obj)), obj.encode("utf-8")))
        self.addNull()

        return self.con.tell()

    def addCString(self, obj):
        self.con.write(struct.pack("{0}s".format(len(obj)), obj.encode("utf-8")))
        self.addNull()
        return self.con.tell()

    def addInteger(self, obj):
        self.addType(spec.INTEGER_TYPE)
        self.con.write(struct.pack("<i", obj))
        return self.con.tell()

    def addDouble(self, obj):
        self.addType(spec.DOUBLE_TYPE)
        self.con.write(struct.pack("<d", obj))
        return self.con.tell()

    def addBool(self, obj):
        self.addType(spec.BOOL_TYPE)
        self.con.write(struct.pack("<B", obj))
        return self.con.tell()

    # Basic list
    def addListHead(self, l):
        self.addType(spec.LIST_TYPE)
        self.addLength(len(l))


    # Basic map
    def addMapHead(self, m):
        self.addType(spec.MAP_TYPE)
        self.addLength(len(m))


    
    # Integer lists
    def addIntegerListHead(self, obj):
        # __istype(self, obj, typeList):
        dtype = self.__listdtype(obj,[float, np.float32, np.float64, int, np.int8, np.int16, 
                    np.int32, np.int64, np.uint, np.uint8, 
                    np.uint16, np.uint32, np.uint64] )



        if dtype == np.dtype("int8"):
            self.addTypedNumList(obj, type=spec.LIST_INT8_TYPE)
            self.numListType = 1
        elif dtype == np.dtype("uint8"):
            self.addTypedNumList(obj, type=spec.LIST_UINT8_TYPE)
        elif dtype == np.dtype("int16"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT16_TYPE)
        elif dtype == np.dtype("uint16"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT16_TYPE)
        elif dtype == np.dtype("int32")  or dtype == int:
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT32_TYPE)
        elif dtype == np.dtype("uint32"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT32_TYPE)
        elif dtype == np.dtype("int64"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT64_TYPE)
        elif dtype == np.dtype("uint64"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT64_TYPE)
        elif dtype == np.dtype("float32"):
            self.numListType = 2
            self.addTypedNumList(obj, type=spec.LIST_FLOAT32_TYPE)
        elif dtype == np.dtype("float64") or dtype == float:
            self.numListType = 2
            self.addTypedNumList(obj, type=spec.LIST_FLOAT64_TYPE)
        else:
            raise ValueError("List type " + str(dtype) + " not found.")

    def addTypedNumList(self, obj, type):
        _l = len(obj)
        self.addType(type)
        self.addLength(_l)

        # self.con.write(obj.tobytes())



    
    # @profile
    def addChunkedNumericArray(self, obj, chunkSize, currentWritten=0, startIndex=0):
        bytesWritten = currentWritten

        idx = startIndex
        lObj = len(obj)

        isBaseList = type(obj) == list
        if not isBaseList:
            while idx < lObj:
                bytesWritten = bytesWritten + self.con.write(obj[idx].tobytes())
                idx = idx + 1
                if bytesWritten >= chunkSize:
                    break
        else:
            while idx < lObj and bytesWritten < chunkSize:
                # Base python list
                if self.numListType == 1:
                    bytesWritten = bytesWritten + self.con.write(struct.pack("<i", obj[idx]))
                else:
                    bytesWritten = bytesWritten + self.con.write(struct.pack("<d", obj[idx]))
                idx = idx + 1

            
        if idx >= len(obj):
            idx = -1
        return [self.con.tell(), idx]

    def addChunkedStringArray(self, obj, chunkSize, currentWritten=0, startIndex=0):
        bytesWritten = currentWritten
        idx = startIndex
        lObj = len(obj)
        while idx < lObj:
            nB1 = self.getSize()
            nB2 = self.addCString(obj[idx])

            bytesWritten = bytesWritten + (nB2-nB1)
            idx = idx + 1


            if bytesWritten >= chunkSize:
                break

        if idx >= len(obj):
            idx = -1
        return [self.con.tell(), idx]



    def addStringListHead(self, obj):
        count_bytes = 0
        for my_str in obj:
            count_bytes = count_bytes + len(my_str.encode("utf-8"))
        self.addType(spec.LIST_STRING_TYPE)
        self.addLength(count_bytes + len(obj))


    def getBytes(self):
        return self.con.getvalue()

    def clear(self):
        self.con.truncate(0)
        self.con.seek(0)

    def getSize(self):
        return self.con.tell()

class SerializerJsonIterator:
    def __init__(self, jsonData, chunkSize=8*1024):
        self.isMainDict = True
        self.jsonData = list([list(jsonData.values())])

        self.keys = list([list(jsonData.keys())])
        self.numObjs = {}

        self.buffer = io.BytesIO
        self.currentKey = 0

        self.longArray = False
        self.level = 0
        self.arrayIdx = [0] # Has to be array so that we handle lists within generic lists
        self.addingIntegerArray = False
        self.addingStringArray = False

        self.chunkedIndex = 0
        
        self.array = None

      
        
        self.serializer = SerializerIt()
        self.serializer.addTsonSpec()
        self.serializer.addMapHead(jsonData)

        self.maxChunk = chunkSize
        self.bytesWritten = 0


    # @profile
    def __listtype(self, obj):
        currType = None
        prevType = None

        if len(obj) == 0:
            return MIXED_LIST        
        elif isinstance(obj[0], str):
            listType = STRING_LIST
        elif isinstance(obj[0], (int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64, float,  np.float32, np.float64)):
            listType = NUMERIC_LIST
        else:
            return MIXED_LIST

        baseClass = obj[0].__class__
        z = [isinstance(o, baseClass) != baseClass for o in obj]
        classComp = list(compress(range(len(z)), z))

        if len(classComp) > 0:
            if listType == NUMERIC_LIST and isinstance(obj[classComp[0]], (int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64, float,  np.float32, np.float64)):
                return MIXED_NUMERIC_LIST
            else:
                return MIXED_LIST

        
        return listType
            



    def isAddingArray(self):
        return self.addingIntegerArray or self.addingStringArray

    def __iter__(self):
        return self

    # @profile
    def __next__(self):
         
        while True:
            notAddingArray = self.addingIntegerArray == False and self.addingStringArray == False
            if notAddingArray:

                if not self.level in self.numObjs:
                    self.numObjs[self.level] = len(self.keys[self.level][:])
                
                n = self.numObjs[self.level]
                idx = self.arrayIdx[self.level]

                while idx >= n:
                    self.level= self.level -1
                    if self.level <= -1:
                        break

                    if not self.level in self.numObjs:
                        self.numObjs[self.level] = len(self.keys[self.level][:])
                    
                    n = self.numObjs[self.level]
                    
                    
                    idx = self.arrayIdx[self.level]

                if self.level <= -1: 
                    if self.serializer.getSize() > 0:
                        bts = self.serializer.getBytes()
                        self.serializer.clear()
                        return bts
                    else:
                        raise StopIteration
                
                self.arrayIdx[self.level] = self.arrayIdx[self.level] + 1

                key = self.keys[self.level][idx]


                if key != "@@NOKEY@@":
                    self.serializer.addString(key)

                obj = self.jsonData[self.level][idx]


            else:
                obj = self.array
             
            if obj is None and notAddingArray:
                self.bytesWritten = self.bytesWritten + self.serializer.addNull()
            elif isinstance(obj, bool ) and notAddingArray:
                self.bytesWritten = self.bytesWritten + self.serializer.addBool(obj)
            elif isinstance(obj, str) and notAddingArray:
                self.bytesWritten = self.bytesWritten + self.serializer.addString(obj)

            elif isinstance(obj, (float,  np.float32, np.float64)) and notAddingArray:
                self.bytesWritten = self.bytesWritten + self.serializer.addDouble(obj)
            elif isinstance(obj, (int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)) and notAddingArray:
                self.bytesWritten = self.bytesWritten + self.serializer.addInteger(obj)

            # # # String, Int/float and other lists
            elif isinstance(obj, np.ndarray) or isinstance(obj, list):
                if notAddingArray:
                    listType = self.__listtype(obj)
                else:
                    listType = None
                # if self.__islisttype(obj, typeList=[str]):
                if self.addingStringArray or listType == STRING_LIST:
                    if not self.addingStringArray:
                        self.serializer.addStringListHead(obj)

                    
                    res = self.serializer.addChunkedStringArray(obj,  self.maxChunk, self.bytesWritten, self.chunkedIndex)
                    self.bytesWritten = self.bytesWritten + res[0]
                    # obj = res[1]
                    
                    

                    if res[1] >= 0:
                        # There is still more to add
                        self.addingStringArray = True
                        self.array = obj
                        self.chunkedIndex = res[1]
                    else:
                        self.addingStringArray = False
                        self.array = None
                        self.chunkedIndex = 0

                elif self.addingIntegerArray or listType == NUMERIC_LIST or listType == MIXED_NUMERIC_LIST:
                    # if isinstance(obj, list):
                        # Converts
                        # pass
                        # raise TsonError("Base Python lists are not supported for numeric arrays. Please convert to numpy array with numpy.array(list).")
                    if not self.addingIntegerArray:
                        if listType == MIXED_NUMERIC_LIST:
                            # Upcasts the list to float
                            obj = [float(i) for i in obj]
                        self.serializer.addIntegerListHead(obj)

                    

                    res = self.serializer.addChunkedNumericArray(obj,  self.maxChunk, self.bytesWritten, self.chunkedIndex)
                    self.bytesWritten = self.bytesWritten + res[0]

                    if res[1] >= 0:
                        # There is still more to add
                        self.addingIntegerArray = True
                        self.array = obj
                        self.chunkedIndex = res[1]
                    else:
                        self.addingIntegerArray = False
                        self.array = None
                        self.chunkedIndex = 0


                else:
                    self.serializer.addListHead(obj)

                    
                    if not self.level in self.numObjs:
                        self.numObjs[self.level] = 0

                    for k in np.arange(len(obj)-1, -1, -1):
                        o = obj[k]
                        self.jsonData[self.level].insert(self.arrayIdx[self.level], o)
                        self.keys[self.level].insert(self.arrayIdx[self.level],"@@NOKEY@@")
                        self.numObjs[self.level] = self.numObjs[self.level]+1

                    # self.numObjs[self.level] = len(self.keys[self.level][:])

            # # Maps
            elif isinstance(obj, dict):
                self.level = self.level + 1

                if self.level >= len(self.arrayIdx):
                    self.arrayIdx.append(0)
                else:
                    # Level was previously reached, so we overwrite the data
                    self.arrayIdx[self.level] = 0

                self.serializer.addMapHead(obj)
                keys = list(obj.keys())

                if self.level >= len(self.keys):
                    self.keys.append(keys)
                    self.jsonData.append(list(obj.values()))
                else:
                    self.keys[self.level] = keys
                    self.numObjs[self.level] = len(keys)
                    self.jsonData[self.level] = list(obj.values())

                continue

            else:
                raise TsonError("Unknown object type.")



            if self.bytesWritten   > self.maxChunk:
                bts = self.serializer.getBytes()
                self.serializer.clear()
                self.bytesWritten = 0
                return bts
            else:
                continue

class Serializer:
    def __init__(self, obj, con=None):
        self.con = con or StringIO()

        self.addString(spec.TSON_SPEC_VERSION)
        self.addObject(obj)


    def addType(self, spec_type):
        self.con.write(struct.pack("<B", spec_type))

    def addLength(self, length):
        # to:do - check list length
        self.con.write(struct.pack("<I", length))


    # Add object
    def __istype(self, obj, typeList):
        return np.any(  [isinstance(obj, t) for t in typeList ] )

    def __islisttype(self, obj, typeList):
        res = True

        if len(obj) == 0:
            res = False
        else:
            res = all([isinstance(o,tuple(typeList)) for o in obj])

        return res


    def __listdtype(self, obj, typeList):
        # Assumptions
        #  1. List has been checked to ensure all ae of the same type
        #  2. Values will only match one type within typeList
        o = obj[0]

        typeCheck = [isinstance(o, t) for t in typeList ]

        res = [i for i, val in enumerate(typeCheck) if val]
        return typeList[res[0]]
    
    def __listtype(self, obj):
        currType = None
        prevType = None

        if len(obj) == 0:
            return MIXED_LIST        
        elif isinstance(obj[0], str):
            listType = STRING_LIST
        elif isinstance(obj[0], (int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64, float,  np.float32, np.float64)):
            listType = NUMERIC_LIST
        else:
            return MIXED_LIST

            
        for o in obj:
            if prevType is None:
                prevType = o.__class__
                continue
            
            currType = o.__class__

            if prevType != currType:
                if listType == NUMERIC_LIST and isinstance(o, (int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64, float,  np.float32, np.float64)):
                    return MIXED_NUMERIC_LIST
                else:
                    return MIXED_LIST
        
        return listType

            
        #'test', False, '2035'
    def addObject(self, obj):
        # Basic types
        # bool must come before int

        # If not yet instantiated, do so
        # if isinstance(obj, types.GeneratorType):
            # obj = next(obj)


        if obj is None:
            self.addNull()
        elif isinstance(obj, bool):
            self.addBool(obj)
        elif self.__istype(obj, typeList=[str]):
            self.addString(obj)

        elif self.__istype(obj, typeList=[float,  np.float32, np.float64]):
            self.addDouble(obj)
        elif self.__istype(obj, typeList=[int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64]):
            self.addInteger(obj)

        # # String, Int/float and other lists
        elif isinstance(obj, np.ndarray) or isinstance(obj, list):
            listType = self.__listtype(obj)

            if listType == STRING_LIST:#self.__islisttype(obj, typeList=[str]):
                self.addStringList(obj)
            elif listType == NUMERIC_LIST or listType == MIXED_NUMERIC_LIST:#self.__islisttype(obj, typeList=[float, np.float32, np.float64, int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64]):
                # if isinstance(obj, list):
                if listType == MIXED_NUMERIC_LIST:
                    # Upcasts the list to float
                    obj = [float(i) for i in obj]
                    # raise TsonError("Base Python lists are not supported for numeric arrays. Please convert to numpy array with numpy.array(list).")
                self.addIntegerList(obj)
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
            if  not k is None and not (isinstance(k, str)) and len(k) != 0:
                raise TsonError("Map key must be a String.")

            self.addString(k)
            self.addObject(v)

    def addMapThreaded(self, m):
        self.addType(spec.MAP_TYPE)
        self.addLength(len(m))

        for k, v in m.items():
            if  not k is None and not (isinstance(k, str)) and len(k) != 0:
                raise TsonError("Map key must be a String.")

            self.addString(k)
            self.addObjectThreaded(v)
    
    
    # Integer lists
    def addIntegerList(self, obj):
        # __istype(self, obj, typeList):
        dtype = self.__listdtype(obj,[float, np.float32, np.float64, int, np.int8, np.int16, 
                    np.int32, np.int64, np.uint, np.uint8, 
                    np.uint16, np.uint32, np.uint64] )



        if dtype == np.dtype("int8"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT8_TYPE)
            
        elif dtype == np.dtype("uint8"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT8_TYPE)
            
        elif dtype == np.dtype("int16"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT16_TYPE)
            
        elif dtype == np.dtype("uint16"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT16_TYPE)
            
        elif dtype == np.dtype("int32")  or dtype == int:
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT32_TYPE)
            
        elif dtype == np.dtype("uint32"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT32_TYPE)
            
        elif dtype == np.dtype("int64"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT64_TYPE)
            
        elif dtype == np.dtype("uint64"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT64_TYPE)
        elif dtype == np.dtype("float32"):
            self.numListType = 2
            self.addTypedNumList(obj, type=spec.LIST_FLOAT32_TYPE)
        elif dtype == np.dtype("float64") or dtype == float:
            self.numListType = 2
            self.addTypedNumList(obj, type=spec.LIST_FLOAT64_TYPE)
        else:
            raise ValueError("List type " + str(dtype) + " not found.")

    def addTypedNumList(self, obj, type):
        _l = len(obj)
        self.addType(type)
        self.addLength(_l)


        try:
            # bytesWritten = bytesWritten + self.con.write(obj[idx].tobytes())
            self.con.write(obj.tobytes())
        except:
            # Base python list
            if self.numListType == 1:
                for o in obj:
                    self.con.write(struct.pack("<i", o))
                # bytesWritten = bytesWritten + self.con.tell()
            else:
                for o in obj:
                    self.con.write(struct.pack("<d", o))
        # self.con.write(obj.tobytes())




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
