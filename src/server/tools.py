import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from braintrust import traced
from server.chart_models import create_chart
from server.client import get_client
from server.models import QueryParameters
from server.vectorstore import SemanticLayerVectorStore

logger = logging.getLogger(__name__)


class SemanticLayerSearchInput(BaseModel):
    query: str = Field(
        description="The natural language query to search for metrics and dimensions"
    )
    k_metrics: int = Field(
        default=5, description="Number of metrics to retrieve (default: 5)"
    )
    k_dimensions: int = Field(
        default=5,
        description="Number of dimensions to retrieve per metric (default: 5)",
    )


class SemanticLayerSearchTool(BaseTool):
    name: str = "semantic_layer_metadata"
    description: str = """
    Search for relevant metrics and dimensions in the semantic layer based on a natural language query.
    Use this tool when you need to find metrics and dimensions that match what the user is asking about.
    """
    args_schema: type[BaseModel] = SemanticLayerSearchInput
    vector_store: SemanticLayerVectorStore = Field(
        description="The vector store for semantic layer metadata search"
    )

    def __init__(self, vector_store: SemanticLayerVectorStore, **kwargs):
        super().__init__(vector_store=vector_store, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous run is not supported, use arun instead."""
        raise NotImplementedError("This tool only supports async operations")

    @traced(name="semantic_layer_metadata", type="tool")
    async def _arun(
        self, query: str, k_metrics: int = 5, k_dimensions: int = 5
    ) -> dict[str, Any]:
        """Run the metric search."""
        try:
            logger.debug(f"Searching for metrics and dimensions with query: {query}")

            # Validate k_metrics and k_dimensions are positive integers
            k_metrics = max(1, min(k_metrics, 5))  # Limit between 1 and 20
            k_dimensions = max(1, min(k_dimensions, 5))

            result = await self.vector_store.retrieve(
                query=query, k_metrics=k_metrics, k_dimensions=k_dimensions
            )

            logger.debug(
                f"Found {len(result.metrics)} metrics and {len(result.dimensions)} dimensions"
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f"Error in semantic layer search: {e}")
            # Return a more graceful error response instead of raising
            return {"metrics": [], "dimensions": [], "query": query, "error": str(e)}


class SemanticLayerQueryTool(BaseTool):
    name: str = "semantic_layer_query"
    description: str = """
    Query the semantic layer for metrics and dimensions.
    IMPORTANT: You should ALWAYS use the semantic_layer_metadata tool first to find available metrics and dimensions.
    Then use this tool to query the data using only the metrics and dimensions returned by semantic_layer_metadata.

    Parameters:
    - metrics (required): List of metric names from search results
    - group_by (optional): List of dimension names for grouping
    - where (optional): List of filter conditions using TimeDimension() or Dimension() templates
    - order_by (optional): List of ordering specs for metrics or dimensions
    - limit (optional): Number of results to return

    Do not make up metrics or dimensions, only use those returned by the semantic_layer_metadata tool.
    """
    args_schema: type[BaseModel] = QueryParameters
    return_direct: bool = True  # This ensures the response goes directly to the model

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous run is not supported, use arun instead."""
        raise NotImplementedError("This tool only supports async operations")

    def _format_data(self, data: dict) -> dict:
        """Format data to be JSON serializable and properly formatted."""
        formatted_data = {}
        for key, values in data.items():
            formatted_values = []
            for value in values:
                if hasattr(value, "isoformat"):  # Handle datetime objects
                    # Keep datetime display format for table view
                    formatted_values.append(value.strftime("%Y-%m-%d"))
                elif str(type(value).__name__) == "Decimal":
                    formatted_values.append(float(value))
                else:
                    formatted_values.append(value)
            formatted_data[key] = formatted_values
        return formatted_data

    @traced(name="semantic_layer_query", type="tool")
    async def _arun(
        self,
        metrics: list[str],
        group_by: list[str] | None = None,
        limit: int | None = None,
        order_by: list[str] | None = None,
        where: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query the semantic layer to return data requested by the user."""
        try:
            # Create a fresh client for each query
            client = get_client()

            # Ensure defaults for all fields
            group_by = group_by or []
            order_by = order_by or []
            where = where or []

            logger.debug(
                f"Querying semantic layer with metrics: {metrics}, "
                f"group_by: {group_by}, limit: {limit}, "
                f"order_by: {order_by}, where: {where}"
            )

            async with client.session():
                # Execute query and get SQL concurrently
                table, sql = await asyncio.gather(
                    client.query(
                        metrics=metrics,
                        group_by=group_by,
                        limit=limit,
                        order_by=order_by,
                        where=where,
                    ),
                    client.compile_sql(
                        metrics=metrics,
                        group_by=group_by,
                        limit=limit,
                        order_by=order_by,
                        where=where,
                    ),
                )

                logger.debug("Query completed successfully")

                chart_js_config = None
                try:
                    chart = create_chart(
                        metrics=metrics, dimensions=group_by, table=table
                    )
                    chart_js_config = chart.get_config()
                except Exception as e:
                    logger.error(f"Error creating chart config: {e}")
                    # Provide a fallback chart configuration that shows an error message
                    chart_js_config = {
                        "type": "bar",  # Use simple bar chart as fallback
                        "data": {"labels": [], "datasets": []},
                        "options": {
                            "responsive": True,
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Unable to create chart visualization",
                                    "color": "#94EAD4",  # Using one of our theme colors
                                    "font": {"size": 16},
                                },
                                "subtitle": {
                                    "display": True,
                                    "text": str(e),
                                    "color": "#666",
                                    "font": {"size": 14},
                                },
                            },
                        },
                    }

                # Convert table to dict and format the data
                data_dict = table.to_pydict()
                formatted_data = self._format_data(data_dict)

                # Format the response for the frontend - ensure it's wrapped correctly for direct return
                return {
                    "type": "function_call_output",  # This matches what the frontend expects
                    "output": json.dumps(
                        {
                            "type": "query_result",
                            "sql": sql,
                            "data": formatted_data,
                            "chart_config": chart_js_config,
                            # TODO: Remove this once we're handling metrics in the frontend via query
                            "metrics": metrics,
                            "query": QueryParameters(
                                metrics=metrics,
                                group_by=group_by or [],
                                limit=limit,
                                order_by=order_by or [],
                                where=where or [],
                            ).model_dump(),
                        }
                    ),
                }

        except Exception as e:
            logger.error(f"Error in semantic layer query: {e}")
            return {"error": str(e), "type": "error"}


def create_tools() -> Sequence[BaseTool]:
    """Create the tools with their required dependencies."""
    # Create vector store for this connection
    vector_store = SemanticLayerVectorStore()

    tavily_tool = TavilySearchResults(
        max_results=5,
        include_answer=True,
        description=(
            "This is a search tool for accessing the internet.\n\n"
            "Let the user know you're asking your friend Tavily for help before you call the tool."
        ),
    )

    return [
        SemanticLayerSearchTool(vector_store=vector_store),
        SemanticLayerQueryTool(),
        tavily_tool,
    ]
