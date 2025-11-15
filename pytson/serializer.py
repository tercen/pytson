import struct, math
import numpy as np
import pytson.spec as spec
from pytson.error import TsonError
from collections import deque
import io
from io import BytesIO as StringIO

STRING_LIST = 0
NUMERIC_LIST = 1
MIXED_LIST = 2
MIXED_NUMERIC_LIST = 3


class SerializerIt:
    """Optimized iterator-based serializer with reduced memory footprint"""
    __slots__ = ['con', 'numListType', 'null_pack', 'objLen']
    
    def addTsonSpec(self):
        self.addString(spec.TSON_SPEC_VERSION)

    def __init__(self, con=None):
        self.con = con if con is not None else io.BytesIO()
        self.numListType = None
        self.null_pack = struct.pack("<B", spec.NULL_TYPE)
        self.objLen = None

    def addType(self, spec_type):
        self.con.write(struct.pack("<B", spec_type))

    def addLength(self, length):
        self.con.write(struct.pack("<I", length))

    def __listdtype(self, obj, typeList):
        """Determine the dtype of a list - optimized to check first element only"""
        o = obj[0]
        for dtype in typeList:
            if isinstance(o, dtype):
                return dtype
        return None

    # Basic types (null, string, integer, double, bool)
    def addNull(self):
        self.addType(spec.NULL_TYPE)
        return self.con.tell()

    def addString(self, obj):
        self.addType(spec.STRING_TYPE)
        # Encode once and write directly
        encoded = obj.encode("utf-8")
        self.con.write(struct.pack(f"{len(encoded)}s", encoded))
        self.con.write(self.null_pack)
        return self.con.tell()

    def addCString(self, obj):
        """Optimized C-string writing - works with iterators to avoid copies"""
        if not obj:
            return
        
        # Accept both lists and iterators
        # Calculate total size needed - if obj is an iterator, we need to materialize it
        # But we can write directly without creating intermediate bytearray
        
        # For small lists, use the optimized bytearray approach
        # For iterators or large lists, write directly to avoid memory spike
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            # Write strings one by one to avoid creating large intermediate buffer
            for s in obj:
                encoded = s.encode("utf-8")
                self.con.write(encoded)
                self.con.write(b'\x00')  # null terminator
        else:
            # Single string
            encoded = obj.encode("utf-8")
            self.con.write(encoded)
            self.con.write(b'\x00')

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
        dtype = self.__listdtype(obj, [float, np.float32, np.float64, int, np.int8, np.int16, 
                    np.int32, np.int64, np.uint, np.uint8, 
                    np.uint16, np.uint32, np.uint64])

        if dtype == np.dtype("int8"):
            self.addTypedNumList(obj, type=spec.LIST_INT8_TYPE)
            self.numListType = 1
        elif dtype == np.dtype("uint8"):
            self.addTypedNumList(obj, type=spec.LIST_UINT8_TYPE)
            self.numListType = 1
        elif dtype == np.dtype("int16"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_INT16_TYPE)
        elif dtype == np.dtype("uint16"):
            self.numListType = 1
            self.addTypedNumList(obj, type=spec.LIST_UINT16_TYPE)
        elif dtype == np.dtype("int32") or dtype == int:
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
        """Optimized typed numeric list - avoid unnecessary conversions"""
        import array
        
        _l = len(obj)
        self.addType(type)
        self.addLength(_l)

        # For numpy arrays, use tobytes() directly - fastest method
        if isinstance(obj, np.ndarray):
            self.con.write(obj.tobytes())
        else:
            # For Python lists, use array module - faster than numpy conversion for large lists
            if self.numListType == 1:
                arr = array.array('i', obj)
            else:
                arr = array.array('d', obj)
            self.con.write(arr.tobytes())

    def addChunkedNumericArray(self, obj, chunkSize, currentWritten=0, startIndex=0):
        """Optimized chunked numeric array writing - ZERO COPY for numpy arrays"""
        import array
        
        bytesWritten = currentWritten
        idx = startIndex
        lObj = len(obj)

        # CRITICAL OPTIMIZATION: For numpy arrays, use array views (no copy!)
        if isinstance(obj, np.ndarray):
            # Calculate how many elements we can write
            if self.numListType == 1:
                itemSize = obj.itemsize  # Use actual item size from numpy
            else:
                itemSize = obj.itemsize
            
            bytesToWrite = chunkSize - currentWritten
            nToWrite = min(bytesToWrite // itemSize, lObj - idx)
            
            if nToWrite > 0:
                # Use numpy array slicing - this creates a VIEW, not a copy!
                # Then tobytes() creates the bytes representation
                view = obj[idx:idx + nToWrite]
                self.con.write(view.tobytes())
                idx += nToWrite
        else:
            # For Python lists, we must use array module
            # But we can minimize copies by writing directly
            bytesToWrite = chunkSize - currentWritten
            
            if self.numListType == 1:
                typeString = "i"
                itemSize = 4
            else:
                typeString = "d"
                itemSize = 8
            
            nToWrite = min(bytesToWrite // itemSize, lObj - idx)
            
            if nToWrite > 0:
                # OPTIMIZATION: Create array.array directly from iterator to avoid list slice
                # Original: array.array(typeString, obj[idx:idx+nToWrite]) - creates slice copy!
                # Optimized: Use itertools.islice for zero-copy iteration
                from itertools import islice
                arr = array.array(typeString, islice(obj, idx, idx + nToWrite))
                self.con.write(arr.tobytes())
                idx += nToWrite

        if idx >= len(obj):
            idx = -1
        return [self.con.tell(), idx]

    def addChunkedStringArray(self, obj, chunkSize, currentWritten=0, startIndex=0):
        """Optimized chunked string array writing - zero-copy iteration"""
        bytesWritten = currentWritten
        idx = startIndex
        lObj = len(obj)
        bytesToWrite = chunkSize - currentWritten

        if bytesToWrite > 0:
            i1 = idx
            
            # Calculate how many strings fit in the chunk
            while idx < lObj and bytesWritten < chunkSize:
                str_size = len(obj[idx].encode("utf-8")) + 1  # +1 for null terminator
                if bytesWritten + str_size > chunkSize and idx > i1:
                    break
                bytesWritten += str_size
                idx += 1

            if idx > i1:
                # Use islice to create an iterator - no list copy!
                from itertools import islice
                self.addCString(islice(obj, i1, idx))

        if idx >= len(obj):
            idx = -1

        return [self.con.tell(), idx]

    def addStringListHead(self, obj):
        """Optimized string list header - single pass calculation"""
        # Note: We must encode to count bytes accurately for UTF-8
        # The strings will be encoded again when written, but this is unavoidable
        # The key optimization is doing it in a single pass without creating intermediate lists
        count_bytes = sum(len(s.encode("utf-8")) for s in obj)
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
    """Optimized JSON iterator with reduced memory footprint using deques"""
    __slots__ = ['isMainDict', 'jsonData', 'keys', 'numObjs', 'buffer', 'currentKey',
                 'longArray', 'level', 'arrayIdx', 'addingIntegerArray', 'addingStringArray',
                 'chunkedIndex', 'array', 'serializer', 'maxChunk', 'bytesWritten']
    
    def __init__(self, jsonData, chunkSize=8*1024):
        self.isMainDict = True
        # Use deques for better insertion performance
        self.jsonData = [deque(jsonData.values())]
        self.keys = [deque(jsonData.keys())]
        self.numObjs = {}

        self.buffer = io.BytesIO
        self.currentKey = 0

        self.longArray = False
        self.level = 0
        self.arrayIdx = [0]
        self.addingIntegerArray = False
        self.addingStringArray = False

        self.chunkedIndex = 0
        self.array = None
        
        self.serializer = SerializerIt()
        self.serializer.addTsonSpec()
        self.serializer.addMapHead(jsonData)

        self.maxChunk = chunkSize
        self.bytesWritten = 0

    def __listtype(self, obj):
        """Optimized list type detection with early exit and NaN check optimization"""
        if len(obj) == 0:
            return MIXED_LIST
        
        first = obj[0]
        
        if isinstance(first, str):
            return STRING_LIST
        elif isinstance(first, (int, np.int8, np.int16, np.int32, np.int64, np.uint, 
                               np.uint8, np.uint16, np.uint32, np.uint64, float, 
                               np.float32, np.float64)):
            # Optimized NaN check - only for numeric types
            try:
                # Use numpy's fast NaN detection if it's an array
                if isinstance(obj, np.ndarray):
                    if np.isnan(obj).any():
                        return MIXED_LIST
                else:
                    # For lists, check if any element is NaN
                    if any(isinstance(x, float) and math.isnan(x) for x in obj):
                        return MIXED_LIST
            except (TypeError, ValueError):
                pass
            
            return NUMERIC_LIST
        else:
            return MIXED_LIST

    def isAddingArray(self):
        return self.addingIntegerArray or self.addingStringArray

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            notAddingArray = not self.addingIntegerArray and not self.addingStringArray
            
            if notAddingArray:
                if self.level not in self.numObjs:
                    self.numObjs[self.level] = len(self.keys[self.level])
                
                n = self.numObjs[self.level]
                idx = self.arrayIdx[self.level]

                while idx >= n:
                    self.level -= 1
                    if self.level <= -1:
                        break

                    if self.level not in self.numObjs:
                        self.numObjs[self.level] = len(self.keys[self.level])
                    
                    n = self.numObjs[self.level]
                    idx = self.arrayIdx[self.level]

                if self.level <= -1:
                    if self.serializer.getSize() > 0:
                        bts = self.serializer.getBytes()
                        self.serializer.clear()
                        return bts
                    else:
                        raise StopIteration
                
                self.arrayIdx[self.level] += 1

                key = self.keys[self.level][idx]

                if key != "@@NOKEY@@":
                    self.serializer.addString(key)

                obj = self.jsonData[self.level][idx]
            else:
                obj = self.array

            if obj is None and notAddingArray:
                self.bytesWritten += self.serializer.addNull()
            elif isinstance(obj, bool) and notAddingArray:
                self.bytesWritten += self.serializer.addBool(obj)
            elif isinstance(obj, str) and notAddingArray:
                self.bytesWritten += self.serializer.addString(obj)
            elif isinstance(obj, (float, np.float32, np.float64)) and notAddingArray:
                self.bytesWritten += self.serializer.addDouble(obj)
            elif isinstance(obj, (int, np.int8, np.int16, np.int32, np.int64, np.uint, 
                                 np.uint8, np.uint16, np.uint32, np.uint64)) and notAddingArray:
                self.bytesWritten += self.serializer.addInteger(obj)

            # String, Int/float and other lists
            elif isinstance(obj, (np.ndarray, list)):
                if notAddingArray:
                    listType = self.__listtype(obj)
                else:
                    listType = None
                
                if self.addingStringArray or listType == STRING_LIST:
                    if not self.addingStringArray:
                        self.serializer.addStringListHead(obj)

                    res = self.serializer.addChunkedStringArray(obj, self.maxChunk, 
                                                                self.bytesWritten, self.chunkedIndex)
                    self.bytesWritten += res[0]

                    if res[1] >= 0:
                        self.addingStringArray = True
                        self.array = obj
                        self.chunkedIndex = res[1]
                    else:
                        self.addingStringArray = False
                        self.array = None
                        self.chunkedIndex = 0

                elif self.addingIntegerArray or listType == NUMERIC_LIST or listType == MIXED_NUMERIC_LIST:
                    if not self.addingIntegerArray:
                        # CRITICAL: Do NOT convert numpy arrays to lists - massive memory duplication!
                        # The original code did: obj = obj.tolist() which doubles memory usage
                        # Instead, work with numpy arrays directly
                        
                        # Only convert mixed numeric lists to ensure type consistency
                        if listType == MIXED_NUMERIC_LIST and isinstance(obj, list):
                            # For mixed numeric Python lists, convert to float list
                            obj = [float(x) for x in obj]
                        
                        self.serializer.addIntegerListHead(obj)

                    res = self.serializer.addChunkedNumericArray(obj, self.maxChunk, 
                                                                 self.bytesWritten, self.chunkedIndex)
                    self.bytesWritten += res[0]

                    if res[1] >= 0:
                        self.addingIntegerArray = True
                        self.array = obj
                        self.chunkedIndex = res[1]
                    else:
                        self.addingIntegerArray = False
                        self.array = None
                        self.chunkedIndex = 0

                else:
                    # Mixed list - use deque for efficient insertion
                    self.serializer.addListHead(obj)
                    
                    if self.level not in self.numObjs:
                        self.numObjs[self.level] = 0

                    # Insert in reverse order using deque's efficient appendleft
                    for k in range(len(obj) - 1, -1, -1):
                        o = obj[k]
                        self.jsonData[self.level].insert(self.arrayIdx[self.level], o)
                        self.keys[self.level].insert(self.arrayIdx[self.level], "@@NOKEY@@")
                        self.numObjs[self.level] += 1

            # Maps
            elif isinstance(obj, dict):
                self.level += 1

                if self.level >= len(self.arrayIdx):
                    self.arrayIdx.append(0)
                else:
                    self.arrayIdx[self.level] = 0

                self.serializer.addMapHead(obj)
                keys = list(obj.keys())

                if self.level >= len(self.keys):
                    self.keys.append(deque(keys))
                    self.jsonData.append(deque(obj.values()))
                else:
                    self.keys[self.level] = deque(keys)
                    self.numObjs[self.level] = len(keys)
                    self.jsonData[self.level] = deque(obj.values())

                continue

            else:
                raise TsonError("Unknown object type.")

            if self.bytesWritten > self.maxChunk:
                bts = self.serializer.getBytes()
                self.serializer.clear()
                self.bytesWritten = 0
                return bts
            else:
                continue


class Serializer:
    """Optimized serializer with reduced memory allocations"""
    __slots__ = ['con', 'numListType']
    
    def __init__(self, obj, con=None):
        self.con = con or StringIO()
        self.numListType = None
        self.addString(spec.TSON_SPEC_VERSION)
        self.addObject(obj)

    def addType(self, spec_type):
        self.con.write(struct.pack("<B", spec_type))

    def addLength(self, length):
        self.con.write(struct.pack("<I", length))

    def __istype(self, obj, typeList):
        return any(isinstance(obj, t) for t in typeList)

    def __islisttype(self, obj, typeList):
        if len(obj) == 0:
            return False
        return all(isinstance(o, tuple(typeList)) for o in obj)

    def __listdtype(self, obj, typeList):
        o = obj[0]
        for dtype in typeList:
            if isinstance(o, dtype):
                return dtype
        return None
    
    def __listtype(self, obj):
        """Optimized list type detection"""
        if len(obj) == 0:
            return MIXED_LIST
        
        first = obj[0]
        
        if isinstance(first, str):
            listType = STRING_LIST
        elif isinstance(first, (int, np.int8, np.int16, np.int32, np.int64, np.uint, 
                               np.uint8, np.uint16, np.uint32, np.uint64, float, 
                               np.float32, np.float64)):
            listType = NUMERIC_LIST
        else:
            return MIXED_LIST

        # Check type consistency
        prevType = type(first)
        for o in obj:
            currType = type(o)
            if prevType != currType:
                if listType == NUMERIC_LIST and isinstance(o, (int, np.int8, np.int16, 
                        np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, 
                        np.uint64, float, np.float32, np.float64)):
                    return MIXED_NUMERIC_LIST
                else:
                    return MIXED_LIST
            prevType = currType
        
        return listType

    def addObject(self, obj):
        if obj is None:
            self.addNull()
        elif isinstance(obj, bool):
            self.addBool(obj)
        elif self.__istype(obj, typeList=[str]):
            self.addString(obj)
        elif self.__istype(obj, typeList=[float, np.float32, np.float64]):
            self.addDouble(obj)
        elif self.__istype(obj, typeList=[int, np.int8, np.int16, np.int32, np.int64, 
                                         np.uint, np.uint8, np.uint16, np.uint32, np.uint64]):
            self.addInteger(obj)
        elif isinstance(obj, (np.ndarray, list)):
            listType = self.__listtype(obj)

            if listType == STRING_LIST:
                self.addStringList(obj)
            elif listType == NUMERIC_LIST or listType == MIXED_NUMERIC_LIST:
                if listType == MIXED_NUMERIC_LIST and isinstance(obj, list):
                    # Convert to numpy array for efficiency
                    obj = np.array([float(i) for i in obj], dtype=np.float64)
                self.addIntegerList(obj)
            else:
                self.addList(obj)
        elif isinstance(obj, dict):
            self.addMap(obj)
        else:
            raise TsonError("Unknown object type.")

    # Basic types
    def addNull(self):
        self.addType(spec.NULL_TYPE)

    def addString(self, obj):
        self.addType(spec.STRING_TYPE)
        encoded = obj.encode("utf-8")
        self.con.write(struct.pack(f"{len(encoded)}s", encoded))
        self.addNull()

    def addCString(self, obj):
        """Optimized C-string writing - works with iterators to avoid copies"""
        if not obj:
            return
        
        # Accept both lists and iterators
        # Calculate total size needed - if obj is an iterator, we need to materialize it
        # But we can write directly without creating intermediate bytearray
        
        # For small lists, use the optimized bytearray approach
        # For iterators or large lists, write directly to avoid memory spike
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            # Write strings one by one to avoid creating large intermediate buffer
            for s in obj:
                encoded = s.encode("utf-8")
                self.con.write(encoded)
                self.con.write(b'\x00')  # null terminator
        else:
            # Single string
            encoded = obj.encode("utf-8")
            self.con.write(encoded)
            self.con.write(b'\x00')

    def addInteger(self, obj):
        self.addType(spec.INTEGER_TYPE)
        self.con.write(struct.pack("<i", obj))

    def addDouble(self, obj):
        self.addType(spec.DOUBLE_TYPE)
        self.con.write(struct.pack("<d", obj))

    def addBool(self, obj):
        self.addType(spec.BOOL_TYPE)
        self.con.write(struct.pack("<B", obj))

    def addList(self, l):
        self.addType(spec.LIST_TYPE)
        self.addLength(len(l))
        for o in l:
            self.addObject(o)

    def addMap(self, m):
        self.addType(spec.MAP_TYPE)
        self.addLength(len(m))
        
        for k, v in m.items():
            if not k is None and not isinstance(k, str) and len(k) != 0:
                raise TsonError("Map key must be a String.")
            self.addString(k)
            self.addObject(v)
    
    def addIntegerList(self, obj):
        """Optimized integer list serialization"""
        dtype = self.__listdtype(obj, [float, np.float32, np.float64, int, np.int8, np.int16, 
                    np.int32, np.int64, np.uint, np.uint8, 
                    np.uint16, np.uint32, np.uint64])

        # Convert numpy scalar types to numpy dtypes for comparison
        if dtype == np.int8 or dtype == np.dtype("int8"):
            list_type, num_type = spec.LIST_INT8_TYPE, 1
        elif dtype == np.uint8 or dtype == np.dtype("uint8"):
            list_type, num_type = spec.LIST_UINT8_TYPE, 1
        elif dtype == np.int16 or dtype == np.dtype("int16"):
            list_type, num_type = spec.LIST_INT16_TYPE, 1
            list_type, num_type = spec.LIST_UINT16_TYPE, 1
        elif dtype == np.int32 or dtype == np.dtype("int32") or dtype == int:
            list_type, num_type = spec.LIST_INT32_TYPE, 1
        elif dtype == np.uint32 or dtype == np.dtype("uint32"):
            list_type, num_type = spec.LIST_UINT32_TYPE, 1
        elif dtype == np.int64 or dtype == np.dtype("int64"):
            list_type, num_type = spec.LIST_INT64_TYPE, 1
        elif dtype == np.uint64 or dtype == np.dtype("uint64"):
            list_type, num_type = spec.LIST_UINT64_TYPE, 1
        elif dtype == np.float32 or dtype == np.dtype("float32"):
            list_type, num_type = spec.LIST_FLOAT32_TYPE, 2
        elif dtype == np.float64 or dtype == np.dtype("float64") or dtype == float:
            list_type, num_type = spec.LIST_FLOAT64_TYPE, 2
        else:
            raise ValueError(f"List type {dtype} not found.")
        
        self.numListType = num_type
        self.addTypedNumList(obj, type=list_type)

    def addTypedNumList(self, obj, type):
        """Optimized typed numeric list - avoid unnecessary conversions"""
        import array
        
        _l = len(obj)
        self.addType(type)
        self.addLength(_l)

        # For numpy arrays, use tobytes() directly - fastest method
        if isinstance(obj, np.ndarray):
            self.con.write(obj.tobytes())
        else:
            # For Python lists, use array module - faster than numpy conversion for large lists
            if self.numListType == 1:
                arr = array.array('i', obj)
            else:
                arr = array.array('d', obj)
            self.con.write(arr.tobytes())

    def addStringList(self, obj):
        """Optimized string list serialization"""
        count_bytes = sum(len(s.encode("utf-8")) for s in obj)
        self.addType(spec.LIST_STRING_TYPE)
        self.addLength(count_bytes + len(obj))
        for x in obj:
            self.addCString(x)

    def getBytes(self):
        return self.con
