from dbtsl.asyncio import AsyncSemanticLayerClient
from dbtsl.models import Metric
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from server.models import (
    RetrievalDimension,
    RetrievalMetric,
    RetrievalResult,
    VectorStoreDimension,
    VectorStoreMetric,
)
from server.settings import settings


class SemanticLayerVectorStore:
    """Vector store for semantic layer metadata."""

    COLLECTION_NAMES = ["metrics", "dimensions"]

    def __init__(self, client: AsyncSemanticLayerClient):
        self.client = client
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = settings.vector_store_path
        self.metric_store = Chroma(
            collection_name="metrics",
            embedding_function=self.embeddings,
            persist_directory=f"{self.persist_directory}/metrics",
        )
        self.dimension_store = Chroma(
            collection_name="dimensions",
            embedding_function=self.embeddings,
            persist_directory=f"{self.persist_directory}/dimensions",
        )

    async def get_metrics(self) -> list[Metric]:
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
        # Get relevant metrics
        metric_docs = self.metric_store.similarity_search(query, k=k_metrics)

        # Convert to simplified metric models
        metrics = [
            RetrievalMetric(
                name=doc.metadata["name"],
                label=doc.metadata["label"] or doc.metadata["name"],
                description=doc.metadata["description"],
                metric_type=doc.metadata["metric_type"],
                requires_metric_time=doc.metadata["requires_metric_time"],
            )
            for doc in metric_docs
        ]
        metric_ids = [m.name for m in metrics]

        # Get relevant dimensions, filtered by retrieved metrics
        dimension_docs = self.dimension_store.similarity_search(
            query, k=k_dimensions, filter={"metric_id": {"$in": metric_ids}}
        )

        # Convert to simplified dimension models
        dimensions = [
            RetrievalDimension(
                name=doc.metadata["name"],
                label=doc.metadata["label"] or doc.metadata["name"],
                description=doc.metadata["description"],
                metric_id=doc.metadata["metric_id"],
            )
            for doc in dimension_docs
        ]

        return RetrievalResult(metrics=metrics, dimensions=dimensions, query=query)

    async def refresh_stores(self) -> None:
        """Refresh the vector stores with latest metadata from the semantic layer."""
        metrics = await self.client.metrics()

        self.metric_store.delete_collection()
        self.dimension_store.delete_collection()

        metric_docs = []
        dimension_docs = []

        for metric in metrics:
            metric_model = VectorStoreMetric(
                name=metric.name,
                label=metric.label,
                description=metric.description,
                metric_type=metric.type,
                requires_metric_time=metric.requires_metric_time,
                dimensions=", ".join([d.name for d in metric.dimensions]),
                queryable_granularities=", ".join(metric.queryable_granularities),
            )
            metadata = metric_model.model_dump()
            metric_docs.append(
                Document(page_content=metric_model.page_content, metadata=metadata)
            )

            for dimension in metric.dimensions:
                dimension_model = VectorStoreDimension(
                    name=dimension.name,
                    dimension_type=dimension.type,
                    qualified_name=dimension.qualified_name,
                    metric_id=metric.name,
                    label=dimension.label,
                    description=dimension.description,
                    expr=dimension.expr,
                )
                metadata = dimension_model.model_dump()
                dimension_docs.append(
                    Document(
                        page_content=dimension_model.page_content, metadata=metadata
                    )
                )

        # Recreate collections with new documents
        self.metric_store = Chroma(
            collection_name="metrics",
            embedding_function=self.embeddings,
            persist_directory=f"{self.persist_directory}/metrics",
        )
        self.dimension_store = Chroma(
            collection_name="dimensions",
            embedding_function=self.embeddings,
            persist_directory=f"{self.persist_directory}/dimensions",
        )

        if metric_docs:
            self.metric_store.add_documents(metric_docs)
        if dimension_docs:
            self.dimension_store.add_documents(dimension_docs)
