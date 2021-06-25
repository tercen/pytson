import os
from pytson.deserializer import DeSerializer

f = "pytson/tests/test_data.tson"

tson_data = open(f, "rb")

print(tson_data)
x = DeSerializer(tson_data)
data = x.obj

print(data["940"])
print(data["851"])
