import asyncio
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
from server.chart_models import create_chart
from server.client import get_client
from server.models import Message
from server.prompt import INSTRUCTIONS
from server.storage import ConversationStorage
from server.tools import create_tools
from server.utils import DateTimeEncoder, format_pyarrow_table, websocket_stream
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
        # Clean up vector store
        if hasattr(app.state, "vector_store"):
            try:
                # Ensure we await any async cleanup
                if app.state.vector_store.metric_store:
                    await app.state.vector_store.metric_store.delete()
                if app.state.vector_store.dimension_store:
                    await app.state.vector_store.dimension_store.delete()
                app.state.vector_store = None
                logger.info("Vector store cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up vector store: {e}")

        # Clean up client
        if hasattr(app.state, "client"):
            try:
                await app.state.client.close()
                app.state.client = None
                logger.info("Semantic Layer client cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up client: {e}")

        logger.info("Application cleanup completed")


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


async def update_conversation_context(request):
    """Update a conversation's context."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        data = await request.json()
        storage.update_conversation_context(conversation_id, context=data["context"])
        return JSONResponse({"status": "success"})
    except Exception as e:
        logger.error(f"Error updating conversation context: {e}")
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
            conversation_id=conversation_id,
        )
        message_id = storage.add_message(conversation_id, message)
        return JSONResponse({"id": message_id})
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def update_message(request):
    """Update a message's data in a conversation."""
    try:
        storage: ConversationStorage = request.app.state.storage
        conversation_id = int(request.path_params["conversation_id"])
        message_id = int(request.path_params["message_id"])

        # Get the conversation and message
        conversation = storage.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)

        message = storage.get_message(message_id)
        if not message or message.conversation_id != conversation_id:
            return JSONResponse({"error": "Message not found"}, status_code=404)

        # For refresh operations, we expect the message to have query data
        if not message.data or "query" not in message.data:
            return JSONResponse({"error": "No query data found"}, status_code=400)

        # Extract query parameters
        query_params = message.data["query"]

        # Execute query
        client = request.app.state.client
        table, sql = await asyncio.gather(
            client.query(**query_params),
            client.compile_sql(**query_params),
        )

        # Generate chart configuration using create_chart function
        metrics = query_params.get("metrics", [])
        dimensions = query_params.get("group_by", [])
        chart = create_chart(metrics, dimensions, table)
        chart_js_config = chart.get_config()

        # Update the message's data
        updated_data = {
            "type": "query_result",
            "sql": sql,
            "data": format_pyarrow_table(table),
            "chart_config": chart_js_config,
            "metrics": metrics,
            "query": query_params,
        }

        # Update the message in storage
        message.data = updated_data
        storage.update_message(message_id, message)

        return JSONResponse(updated_data)

    except Exception as e:
        logger.error(f"Error updating message: {e}")
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

    # Get conversation_id from query parameters
    query_params = dict(websocket.query_params)
    conversation_id = (
        int(query_params.get("conversation_id"))
        if "conversation_id" in query_params
        else None
    )

    # Get context if conversation_id exists
    context = None
    if conversation_id is not None:
        storage: ConversationStorage = websocket.app.state.storage
        conversation = storage.get_conversation(conversation_id)
        if conversation and conversation.context:
            context = conversation.context

    # Modify instructions if context exists
    final_instructions = INSTRUCTIONS
    if context:
        logger.info(f"Applying conversation context: {context}")
        final_instructions = (
            f"{INSTRUCTIONS}\n\n"
            f"IMPORTANT: The following context MUST be applied to ALL queries:\n"
            f"<context>{context}</context>\n\n"
            f"You MUST modify EVERY query to incorporate these requirements as shown in the CONTEXT EXAMPLES above.\n"
            f"If you're unsure how to apply any part of the context, ask the user for clarification."
        )

    browser_receive_stream = websocket_stream(websocket)
    tools = create_tools(websocket.app)

    agent = VoiceToTextReactAgent(
        model="gpt-4o-realtime-preview",
        tools=tools,
        instructions=final_instructions,
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
        "/api/conversations/{conversation_id:int}/context",
        update_conversation_context,
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
    Route(
        "/api/conversations/{conversation_id:int}/messages/{message_id:int}",
        update_message,
        methods=["PUT"],
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
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # )
    uvicorn.run(app, host="0.0.0.0", port=3000)
