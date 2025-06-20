"""
Client creation for the Semantic Layer.
"""

import logging
import warnings

from dbtsl.asyncio import AsyncSemanticLayerClient

from server.settings import settings

# Suppress SSL verification warnings
warnings.filterwarnings(
    "ignore", message="SSL is disabled, certificate verify is disabled"
)
logging.getLogger("gql.transport.aiohttp").setLevel(logging.ERROR)


def get_client() -> AsyncSemanticLayerClient:
    """Create a new Semantic Layer client."""
    return AsyncSemanticLayerClient(
        environment_id=settings.sl.environment_id,
        auth_token=settings.sl.token,
        host=settings.sl.host,
    )
