from collections.abc import AsyncIterator
from datetime import datetime
from json import JSONEncoder

from starlette.websockets import WebSocket


class DateTimeEncoder(JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


async def websocket_stream(websocket: WebSocket) -> AsyncIterator[str]:
    """Create an async iterator from a WebSocket connection."""
    try:
        while True:
            yield await websocket.receive_text()
    except Exception:
        pass
