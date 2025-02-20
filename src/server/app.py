import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket

from langchain_openai_voice import VoiceToTextReactAgent
from server.client import get_client
from server.models import Message
from server.prompt import INSTRUCTIONS
from server.storage import ConversationStorage
from server.tools import create_tools
from server.utils import DateTimeEncoder, websocket_stream
from server.vectorstore import SemanticLayerVectorStore

logger = logging.getLogger(__name__)


class JSONResponse(Response):
    """Custom JSON response that handles datetime serialization."""

    media_type = "application/json"

    def render(self, content) -> bytes:
        """Override render to handle datetime objects."""
        return json.dumps(content, cls=DateTimeEncoder).encode("utf-8")


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Initialize application state and manage semantic layer session."""
    try:
        logger.info("Initializing application state...")

        # Initialize the client
        client = get_client()
        app.state.client = client
        logger.info("Semantic Layer client initialized")

        # Initialize conversation storage
        app.state.storage = ConversationStorage()
        logger.info("Conversation storage initialized")

        # Start a global session for the semantic layer client
        async with client.session():
            logger.info("Semantic Layer session started")

            # Create vector store and refresh metadata
            app.state.vector_store = SemanticLayerVectorStore(client=client)
            await app.state.vector_store.refresh_stores()
            logger.info("Vector store created and refreshed")

            yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Cleaning up application state...")
        if hasattr(app.state, "vector_store"):
            try:
                app.state.vector_store.metric_store.delete_collection()
                app.state.vector_store.dimension_store.delete_collection()
                app.state.vector_store = None
                logger.info("Vector store cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up vector store: {e}")

        if hasattr(app.state, "client"):
            app.state.client = None
            logger.info("Semantic Layer client cleanup completed")


async def list_conversations(request):
    """List all conversations."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversations = storage.list_conversations()
        return JSONResponse([conv.model_dump() for conv in conversations])
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def create_conversation(request):
    """Create a new conversation."""
    try:
        storage: ConversationStorage = request.app.state.storage
        data = await request.json()
        conversation = storage.create_conversation(title=data["title"])
        return JSONResponse(conversation.model_dump())
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_conversation(request):
    """Get a conversation by ID."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        conversation = storage.get_conversation(conversation_id)
        if conversation is None:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        return JSONResponse(conversation.model_dump())
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def update_conversation_title(request):
    """Update a conversation's title."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        data = await request.json()
        storage.update_conversation_title(conversation_id, title=data["title"])
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Error updating conversation title: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def add_message(request):
    """Add a message to a conversation."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        data = await request.json()
        message = Message(
            text=data["text"],
            is_user=data["is_user"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data"),
        )
        storage.add_message(conversation_id, message)
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def clear_messages(request):
    """Clear all messages from a conversation."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        storage.clear_messages(conversation_id)
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Error clearing messages: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def delete_conversation(request):
    """Delete a conversation and all its messages."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        storage.delete_conversation(conversation_id)
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    browser_receive_stream = websocket_stream(websocket)

    # Create tools with access to app state
    tools = create_tools(websocket.app)

    agent = VoiceToTextReactAgent(
        model="gpt-4o-realtime-preview",
        tools=tools,
        instructions=INSTRUCTIONS,
    )

    await agent.aconnect(browser_receive_stream, websocket.send_text)


async def homepage(_request):
    with open("src/server/static/index.html") as f:
        html = f.read()
        return HTMLResponse(html)


routes = [
    Route("/", homepage),
    WebSocketRoute("/ws", websocket_endpoint),
    # Conversation management routes
    Route("/api/conversations", list_conversations, methods=["GET"]),
    Route("/api/conversations", create_conversation, methods=["POST"]),
    Route(
        "/api/conversations/{conversation_id:int}", get_conversation, methods=["GET"]
    ),
    Route(
        "/api/conversations/{conversation_id:int}",
        delete_conversation,
        methods=["DELETE"],
    ),
    Route(
        "/api/conversations/{conversation_id:int}/title",
        update_conversation_title,
        methods=["PUT"],
    ),
    Route(
        "/api/conversations/{conversation_id:int}/messages",
        add_message,
        methods=["POST"],
    ),
    Route(
        "/api/conversations/{conversation_id:int}/messages",
        clear_messages,
        methods=["DELETE"],
    ),
]

app = Starlette(
    debug=True,
    routes=routes,
    lifespan=lifespan,
)

app.mount("/", StaticFiles(directory="src/server/static"), name="static")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=3000)
