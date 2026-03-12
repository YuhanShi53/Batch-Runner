"""
JSON codec helpers with optional orjson acceleration.
"""
from __future__ import annotations

import json
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover - optional dependency
    orjson = None


class JsonCodec:
    """Serialize and deserialize JSON with a consistent fast path."""

    def __init__(self):
        self._use_orjson = orjson is not None

    @property
    def backend_name(self) -> str:
        """Return the active JSON backend name."""
        return "orjson" if self._use_orjson else "json"

    def dumps_bytes(self, value: Any) -> bytes:
        """Serialize a Python object to UTF-8 JSON bytes."""
        if self._use_orjson:
            return orjson.dumps(value)

        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def dumps_text(self, value: Any) -> str:
        """Serialize a Python object to a JSON string."""
        if self._use_orjson:
            return orjson.dumps(value).decode("utf-8")

        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def loads(self, value: Any) -> Any:
        """Deserialize JSON bytes or text into Python objects."""
        if self._use_orjson:
            return orjson.loads(value)

        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)


json_codec = JsonCodec()
