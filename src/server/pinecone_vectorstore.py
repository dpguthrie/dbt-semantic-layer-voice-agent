import logging

from dbtsl.models import Metric
from pinecone import PineconeAsyncio

from server.client import get_client
from server.settings import settings

logger = logging.getLogger(__name__)


class PineconeSemanticLayerVectorStore:
    """Vector store for semantic layer metadata using Pinecone."""

    # Constants for index configuration
    VECTOR_DIMENSION = 1536  # OpenAI embeddings dimension

    def __init__(self):
        """Initialize instance variables."""
        self.client = get_client()
        self.pinecone = PineconeAsyncio(api_key=settings.pinecone_api_key)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.pinecone:
            await self.pinecone.close()

    def _format_category_to_string(self, category: dict) -> str:
        """Convert a category dictionary to a string representation.

        Args:
            category: Dictionary containing category metadata

        Returns:
            String representation of the category with key-value pairs
        """

        # Helper function to format a value, handling lists and other types
        def format_value(value):
            if isinstance(value, list):
                return f"[{', '.join(str(v) for v in value)}]"
            return str(value)

        # Convert dictionary items to strings, filtering out empty values
        formatted_items = []
        for key, value in category.items():
            if value:  # Only include non-empty values
                formatted_value = format_value(value)
                if formatted_value:  # Double check the formatted value isn't empty
                    formatted_items.append(f"{key}: {formatted_value}")

        return "; ".join(formatted_items)

    async def get_metrics(self) -> list[Metric]:
        """Get all metrics from the semantic layer."""
        async with self.client.session():
            return await self.client.get_metrics()

    async def initialize(self):
        """Initialize Pinecone client and indexes asynchronously."""
        # Initialize or verify indexes exist
        await self._initialize_indexes()

    async def _initialize_indexes(self):
        """Initialize Pinecone indexes if they don't exist."""
        existing_indexes = await self.pinecone.list_indexes()

        # Create metric index if it doesn't exist
        if settings.pinecone_metric_index_name not in existing_indexes:
            logger.info(f"Creating metric index {settings.pinecone_metric_index_name}")
            await self.pinecone.create_index_for_model(
                name=settings.pinecone_metric_index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "pinecone-sparse-english-v0",
                    "field_map": {"text": "name"},
                    "metric": "dotproduct",
                },
            )

        # Create dimension index if it doesn't exist
        if settings.pinecone_dimension_index_name not in existing_indexes:
            logger.info(
                f"Creating dimension index {settings.pinecone_dimension_index_name}"
            )
            await self.pinecone.create_index_for_model(
                name=settings.pinecone_dimension_index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "pinecone-sparse-english-v0",
                    "field_map": {"text": "name"},
                    "metric": "dotproduct",
                },
            )

    async def indexes_exist(self) -> bool:
        """Check if both indexes exist and have data."""
        try:
            existing_indexes = await self.pinecone.list_indexes()
            existing_index_names = [index.name for index in existing_indexes]
            return (
                settings.pinecone_metric_index_name in existing_index_names
                and settings.pinecone_dimension_index_name in existing_index_names
            )
        except Exception as e:
            logger.error(f"Error checking indexes: {e}")
            return False

    async def refresh_stores(self) -> None:
        """
        Refresh the vector stores with latest metadata from the semantic layer.
        """
        # Get metrics from database - only operation needing the context manager
        metrics = None
        async with self.client.session():
            metrics = await self.client.metrics()

        if not metrics:
            return

        # Get index hosts
        metric_index_desc = await self.pinecone.describe_index(
            settings.pinecone_metric_index_name
        )
        dimension_index_desc = await self.pinecone.describe_index(
            settings.pinecone_dimension_index_name
        )
        logger.info(f"Metric index host: {metric_index_desc.host}")
        logger.info(f"Dimension index host: {dimension_index_desc.host}")

        # Process metrics in batches of 100
        metric_batch = []
        dimension_batch = []
        seen_dimensions = set()  # Track dimensions we've already processed

        async with (
            self.pinecone.IndexAsyncio(host=metric_index_desc.host) as metric_index,
            self.pinecone.IndexAsyncio(
                host=dimension_index_desc.host
            ) as dimension_index,
        ):
            for metric in metrics:
                metric_data = {
                    "_id": f"{metric.name}",
                    "name": metric.name,
                    "type": "metric",
                    "label": metric.label or "",
                    "description": metric.description or "",
                    "metric_type": metric.type,
                    "requires_metric_time": metric.requires_metric_time,
                    "dimensions": [d.name for d in metric.dimensions],
                    "queryable_granularities": metric.queryable_granularities,
                }
                metric_batch.append(metric_data)

                # Process dimensions for this metric
                for dimension in metric.dimensions:
                    # Skip if we've already processed this dimension
                    if dimension.name in seen_dimensions:
                        continue

                    dimension_data = {
                        "_id": f"{dimension.name}",
                        "name": dimension.name,
                        "type": "dimension",
                        "label": dimension.label or "",
                        "description": dimension.description or "",
                        "dimension_type": dimension.type,
                        "qualified_name": dimension.qualified_name,
                        "expr": dimension.expr or "",
                    }
                    dimension_batch.append(dimension_data)
                    seen_dimensions.add(dimension.name)

            # Upsert any remaining items in the batches
            if metric_batch:
                await metric_index.upsert_records(
                    namespace="default", records=metric_batch
                )
            if dimension_batch:
                await metric_index.upsert_records(
                    namespace="default", records=dimension_batch
                )
