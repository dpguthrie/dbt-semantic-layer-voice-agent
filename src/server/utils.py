from collections.abc import AsyncIterator
from datetime import datetime
from json import JSONEncoder

import pyarrow as pa
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


def format_pyarrow_table(table: pa.Table) -> dict:
    """Format a PyArrow table as a dictionary."""
    formatted_data = {}
    data_dict = table.to_pydict()
    for key, values in data_dict.items():
        formatted_values = []
        for value in values:
            if hasattr(value, "isoformat"):  # Handle datetime objects
                # Keep datetime display format for table view
                formatted_values.append(value.strftime("%Y-%m-%d"))
            elif str(type(value).__name__) == "Decimal":
                formatted_values.append(float(value))
            else:
                formatted_values.append(value)
        formatted_data[key] = formatted_values
    return formatted_data
