import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket

from langchain_openai_voice import VoiceToTextReactAgent
from server.client import get_client
from server.prompt import INSTRUCTIONS
from server.tools import create_tools
from server.utils import websocket_stream
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


# catchall route to load files from src/server/static


routes = [Route("/", homepage), WebSocketRoute("/ws", websocket_endpoint)]

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
