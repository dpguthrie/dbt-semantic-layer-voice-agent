import logging
import os
from typing import Any

import braintrust
from dbtsl import SemanticLayerClient
from pinecone import Pinecone
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SemanticLayerMetadata(BaseModel):
    """Components for retrieving relevant metrics and dimensions from Pinecone."""

    query: str


class SemanticLayerQuery(BaseModel):
    """The parameters of a query to the semantic layer."""

    metrics: list[str]
    group_by: list[str] = Field(default_factory=list)
    where: list[str] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    limit: int | None = None


def get_metadata(query: str, k_results: int = 10) -> list[dict[str, Any]]:
    """Run the metric search using Pinecone."""

    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    try:
        logger.debug(f"Searching for metrics and dimensions with query: {query}")

        # Validate k_results is positive integer
        k_results = max(1, min(k_results, 10))

        # Get index hosts
        metric_index_desc = pinecone.describe_index("semantic-metrics")

        # Search metrics
        results = []
        metric_index = pinecone.Index(host=metric_index_desc.host)
        all_results = metric_index.search(
            namespace="default",
            query={"inputs": {"text": query}, "top_k": 10},
            fields=[
                "name",
                "label",
                "type",
                "description",
                "requires_metric_time",
            ],
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 7,
                "rank_fields": ["name"],
            },
        )
        # Convert metric results to RetrievalMetric objects
        for hit in all_results.result.hits:
            results.append(
                {
                    "type": hit["fields"]["type"],
                    "name": hit["fields"]["name"],
                    "label": hit["fields"]["label"],
                    "description": hit["fields"]["description"],
                    "requires_metric_time": hit["fields"].get(
                        "requires_metric_time", False
                    ),
                }
            )

        return results

    except Exception as e:
        logger.error(f"Error in Pinecone semantic search: {e}")
        return []


def semantic_layer_query(
    metrics: list[str],
    group_by: list[str] | None = None,
    limit: int | None = None,
    order_by: list[str] | None = None,
    where: list[str] | None = None,
) -> dict[str, Any]:
    """Query the semantic layer to return data requested by the user."""

    client = SemanticLayerClient(
        environment_id=os.getenv("SL_ENVIRONMENT_ID"),
        auth_token=os.getenv("SL_TOKEN"),
        host=os.getenv("SL_HOST"),
    )

    try:
        # Ensure defaults for all fields
        group_by = group_by or []
        order_by = order_by or []
        where = where or []

        with client.session():
            # Execute query and get SQL concurrently
            sql = client.compile_sql(
                metrics=metrics,
                group_by=group_by,
                limit=limit,
                order_by=order_by,
                where=where,
            )

        return {
            "sql": sql,
            "query": {
                "metrics": metrics,
                "group_by": group_by,
                "limit": limit,
                "order_by": order_by,
                "where": where,
            },
        }

    except Exception as e:
        logger.error(f"Error in semantic layer query: {e}")
        return {"error": str(e), "type": "error"}


project = braintrust.projects.create(name=os.getenv("BRAINTRUST_PROJECT_NAME"))


project.tools.create(
    handler=get_metadata,
    name="Get Metadata",
    slug="voice-agent-get-metadata-tool",
    description="Get metrics and dimensions from the semantic layer",
    parameters=SemanticLayerMetadata,
)

project.tools.create(
    handler=semantic_layer_query,
    name="Semantic Layer Query",
    slug="voice-agent-semantic-layer-query-tool",
    description="Query the semantic layer to return data requested by the user",
    parameters=SemanticLayerQuery,
)
