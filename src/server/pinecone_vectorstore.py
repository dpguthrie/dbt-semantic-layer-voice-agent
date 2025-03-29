from dbtsl.models import Metric
from langchain_openai import OpenAIEmbeddings
from pinecone import PineconeAsyncio, ServerlessSpec

from server.client import get_client
from server.models import (
    RetrievalResult,
)
from server.settings import settings


class PineconeSemanticLayerVectorStore:
    """Vector store for semantic layer metadata using Pinecone."""

    # Constants for index configuration
    VECTOR_DIMENSION = 1536  # OpenAI embeddings dimension

    def __init__(self):
        """Initialize Pinecone vector store with separate indexes for metrics and dimensions."""
        self.embeddings = OpenAIEmbeddings()
        self.client = get_client()
        self.pinecone = None
        self.metric_index = None
        self.dimension_index = None

    async def initialize(self):
        """Initialize Pinecone client and indexes asynchronously."""
        # Initialize Pinecone client
        self.pinecone = PineconeAsyncio(api_key=settings.pinecone_api_key)

        # Initialize or verify indexes exist
        await self._initialize_indexes()

        # Get index instances
        self.metric_index = self.pinecone.Index(settings.pinecone_metric_index_name)
        self.dimension_index = self.pinecone.Index(
            settings.pinecone_dimension_index_name
        )

    async def _initialize_indexes(self):
        """Initialize Pinecone indexes if they don't exist."""
        existing_indexes = await self.pinecone.list_indexes()

        # Create metric index if it doesn't exist
        if settings.pinecone_metric_index_name not in existing_indexes:
            await self.pinecone.create_index(
                name=settings.pinecone_metric_index_name,
                dimension=self.VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        # Create dimension index if it doesn't exist
        if settings.pinecone_dimension_index_name not in existing_indexes:
            await self.pinecone.create_index(
                name=settings.pinecone_dimension_index_name,
                dimension=self.VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.pinecone:
            await self.pinecone.close()

    async def get_metrics(self) -> list[Metric]:
        """Get all metrics from the semantic layer."""
        async with self.client.session():
            return await self.client.get_metrics()

    async def retrieve(
        self, query: str, k_metrics: int = 5, k_dimensions: int = 5
    ) -> RetrievalResult:
        """
        Retrieve relevant metrics and dimensions from the semantic layer.

        Args:
            query: The search query
            k_metrics: Number of metrics to retrieve
            k_dimensions: Number of dimensions to retrieve per metric

        Returns:
            RetrievalResult containing the most relevant metrics and dimensions
        """
        # TODO: Implement Pinecone-specific retrieval logic
        # This will be implemented in the next iteration
        raise NotImplementedError("Retrieval not yet implemented for Pinecone")

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

        # Process metrics in batches of 100
        metric_batch = []
        dimension_batch = []
        seen_dimensions = set()  # Track dimensions we've already processed

        for metric in metrics:
            metric_vector = await self.embeddings.aembed_query(metric.name)

            metric_data = {
                "id": f"metric_{metric.name}",
                "values": metric_vector,
                "metadata": {
                    "name": metric.name,
                    "label": metric.label,
                    "description": metric.description,
                    "metric_type": metric.type,
                    "requires_metric_time": metric.requires_metric_time,
                    "dimensions": [d.name for d in metric.dimensions],
                    "queryable_granularities": metric.queryable_granularities,
                },
            }
            metric_batch.append(metric_data)

            # Process dimensions for this metric
            for dimension in metric.dimensions:
                # Skip if we've already processed this dimension
                if dimension.name in seen_dimensions:
                    continue

                # Create vector for dimension
                dimension_vector = await self.embeddings.aembed_query(dimension.name)

                dimension_data = {
                    "id": f"dimension_{dimension.name}",
                    "values": dimension_vector,
                    "metadata": {
                        "name": dimension.name,
                        "label": dimension.label,
                        "description": dimension.description,
                        "dimension_type": dimension.type,
                        "qualified_name": dimension.qualified_name,
                        "expr": dimension.expr,
                        "metric_id": metric.name,
                    },
                }
                dimension_batch.append(dimension_data)
                seen_dimensions.add(dimension.name)

            # Upsert when batch size reaches 100 or on last item
            if len(metric_batch) >= 100:
                await self.metric_index.upsert(vectors=metric_batch)
                metric_batch = []

            if len(dimension_batch) >= 100:
                await self.dimension_index.upsert(vectors=dimension_batch)
                dimension_batch = []

        # Upsert any remaining items in the batches
        if metric_batch:
            await self.metric_index.upsert(vectors=metric_batch)
        if dimension_batch:
            await self.dimension_index.upsert(vectors=dimension_batch)
