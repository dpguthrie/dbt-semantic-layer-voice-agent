import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
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
    conversations = request.app.state.storage.list_conversations()
    return JSONResponse(
        json.loads(
            json.dumps(
                [conv.model_dump() for conv in conversations], cls=DateTimeEncoder
            )
        )
    )


async def get_conversation(request):
    """Get a conversation by ID."""
    conversation_id = request.path_params["conversation_id"]
    conversation = request.app.state.storage.get_conversation(conversation_id)
    if conversation:
        return JSONResponse(
            json.loads(json.dumps(conversation.model_dump(), cls=DateTimeEncoder))
        )
    return JSONResponse({"error": "Conversation not found"}, status_code=404)


async def create_conversation(request):
    """Create a new conversation."""
    data = await request.json()
    title = data.get("title", "New Conversation")
    conversation = request.app.state.storage.create_conversation(title)
    return JSONResponse(
        json.loads(json.dumps(conversation.model_dump(), cls=DateTimeEncoder))
    )


async def update_conversation(request):
    """Update a conversation's title."""
    conversation_id = request.path_params["conversation_id"]
    data = await request.json()
    title = data.get("title")
    if not title:
        return JSONResponse({"error": "Title is required"}, status_code=400)

    request.app.state.storage.update_conversation_title(conversation_id, title)
    return JSONResponse({"status": "success"})


async def delete_conversation(request):
    """Delete a conversation."""
    conversation_id = request.path_params["conversation_id"]
    request.app.state.storage.delete_conversation(conversation_id)
    return JSONResponse({"status": "success"})


async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time chat."""
    await websocket.accept()

    try:
        # Get conversation ID from query params
        conversation_id = websocket.query_params.get("conversation_id")
        if conversation_id:
            conversation = websocket.app.state.storage.get_conversation(conversation_id)
            if not conversation:
                await websocket.close(code=4000, reason="Conversation not found")
                return
        else:
            # Create a new conversation if no ID provided
            conversation = websocket.app.state.storage.create_conversation(
                "New Conversation"
            )
            conversation_id = conversation.id

        # Create the agent
        agent = VoiceToTextReactAgent(
            tools=create_tools(websocket.app),
            instructions=INSTRUCTIONS,
            model="gpt-4o-realtime-preview",
        )

        async def send_output_chunk(chunk: str):
            """Send output chunk to WebSocket and save to storage."""
            await websocket.send_text(chunk)

            # Parse the chunk and save to storage if it's a message
            try:
                data = json.loads(chunk)
                if data["type"] in ["assistant.response", "user.input"]:
                    message = Message(
                        text=data["text"],
                        is_user=data["type"] == "user.input",
                        timestamp=datetime.now(UTC),
                        data=None,
                    )
                    websocket.app.state.storage.add_message(conversation_id, message)
                elif data["type"] == "function_call_output":
                    # Save function call output with data for charts/tables
                    try:
                        result = json.loads(data["output"])
                        if result.get("type") == "query_result":
                            message = Message(
                                text=data["output"],
                                is_user=False,
                                timestamp=datetime.now(UTC),
                                data={
                                    "sql": result["sql"],
                                    "data": result["data"],
                                    "chart_config": result.get("chart_config"),
                                },
                            )
                            websocket.app.state.storage.add_message(
                                conversation_id, message
                            )
                    except json.JSONDecodeError:
                        pass
            except (json.JSONDecodeError, KeyError):
                pass

        # Connect the agent to the WebSocket stream
        await agent.aconnect(
            input_stream=websocket_stream(websocket),
            send_output_chunk=send_output_chunk,
        )

    except Exception as e:
        logger.exception("Error in WebSocket handler")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close(code=1011, reason=str(e))
    finally:
        if not websocket.client_state.DISCONNECTED:
            await websocket.close()


routes = [
    Route("/api/conversations", endpoint=list_conversations, methods=["GET"]),
    Route("/api/conversations", endpoint=create_conversation, methods=["POST"]),
    Route(
        "/api/conversations/{conversation_id}",
        endpoint=get_conversation,
        methods=["GET"],
    ),
    Route(
        "/api/conversations/{conversation_id}",
        endpoint=update_conversation,
        methods=["PUT"],
    ),
    Route(
        "/api/conversations/{conversation_id}",
        endpoint=delete_conversation,
        methods=["DELETE"],
    ),
    WebSocketRoute("/ws", endpoint=websocket_endpoint),
    Route(
        "/",
        endpoint=lambda r: HTMLResponse(open("src/server/static/index.html").read()),
    ),
    Mount("/static", StaticFiles(directory="src/server/static"), name="static"),
]

app = Starlette(
    debug=True,
    routes=routes,
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
