from .serializer import Serializer
from .deserializer import DeSerializer
import io
from typing import Any


def encodeTSON(obj: Any) -> io.BytesIO:
    return Serializer(obj, io.BytesIO()).getBytes()


def decodeTSON(bytes) -> Any:
    return DeSerializer(bytes).getObject()
