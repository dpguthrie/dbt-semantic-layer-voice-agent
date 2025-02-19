"""
Client creation for the Semantic Layer.
"""

from dbtsl.asyncio import AsyncSemanticLayerClient

from server.settings import settings


def get_client() -> AsyncSemanticLayerClient:
    """Create a new Semantic Layer client."""
    return AsyncSemanticLayerClient(
        environment_id=settings.sl.environment_id,
        auth_token=settings.sl.token,
        host=settings.sl.host,
    )
