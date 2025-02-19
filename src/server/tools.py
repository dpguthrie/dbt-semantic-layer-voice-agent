import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from server.models import (
    QueryParameters,
)

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
    app: Starlette

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous run is not supported, use arun instead."""
        raise NotImplementedError("This tool only supports async operations")

    async def _arun(
        self, query: str, k_metrics: int = 5, k_dimensions: int = 5
    ) -> dict[str, Any]:
        """Run the metric search."""
        try:
            logger.debug(f"Searching for metrics and dimensions with query: {query}")

            # Validate k_metrics and k_dimensions are positive integers
            k_metrics = max(1, min(k_metrics, 5))  # Limit between 1 and 20
            k_dimensions = max(1, min(k_dimensions, 5))

            result = await self.app.state.vector_store.retrieve(
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
    """
    args_schema: type[BaseModel] = QueryParameters
    app: Starlette
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

    def _determine_chart_type(
        self, data: dict, metrics: list[str], group_by: list[str]
    ) -> dict[str, Any]:
        """
        Determines the optimal chart type and configuration based on data characteristics.
        References: https://eazybi.com/blog/data-visualization-and-chart-types

        Heuristics considered:
        - Number of metrics (1 vs multiple)
        - Number of dimensions (0, 1, 2+)
        - Type of dimensions (temporal vs categorical)
        - Number of unique values per dimension
        - Data distribution
        """
        try:
            # Special case: Single aggregate value with no dimensions
            if not group_by:
                # Get the first metric's data
                metric_data = data.get(metrics[0].upper(), [])
                if not metric_data:
                    metric_data = [None]  # Handle empty data case

                return {
                    "type": "bar",
                    "data": {
                        "labels": ["Total"],
                        "datasets": [
                            {
                                "label": metric,
                                "data": [data.get(metric.upper(), [None])[0]],
                                "backgroundColor": f"hsl({hash(metric) % 360}, 70%, 50%, 0.5)",
                                "borderColor": f"hsl({hash(metric) % 360}, 70%, 50%)",
                            }
                            for metric in metrics
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "plugins": {
                            "title": {
                                "display": True,
                                "text": ", ".join(metrics),
                            }
                        },
                        "scales": {"y": {"beginAtZero": True}},
                    },
                }

            # Get the primary dimension (first group_by)
            primary_dim = group_by[0].upper() if group_by else None
            primary_values = data.get(primary_dim, []) if primary_dim else []

            # Detect if any dimension starts with METRIC_TIME
            is_temporal = False
            temporal_dim = None
            if group_by:
                for dim in group_by:
                    dim_upper = dim.upper()
                    if dim_upper.startswith("METRIC_TIME"):
                        is_temporal = True
                        temporal_dim = dim_upper
                        primary_dim = dim_upper
                        primary_values = data.get(primary_dim, [])
                        break

            # Initialize chart config with defaults
            config = {
                "type": "bar",  # Default type
                "data": {"datasets": []},
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{', '.join(metrics)} by {', '.join(group_by) if group_by else 'Value'}",
                        }
                    },
                },
            }

            # Count unique values in primary dimension
            unique_values = (
                len({str(v) for v in primary_values}) if primary_values else 0
            )

            # Case 2: Temporal dimension
            if is_temporal:
                config["type"] = "line"

                # Convert dates to timestamps for Chart.js
                timestamps = []
                for date_str in primary_values:
                    try:
                        # Parse the date string and convert to milliseconds timestamp
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        timestamp = int(
                            date_obj.timestamp() * 1000
                        )  # Convert to milliseconds
                        timestamps.append(timestamp)
                    except Exception as e:
                        logger.error(f"Error converting date: {e}")
                        timestamps.append(
                            date_str
                        )  # Fallback to original string if parsing fails

                if len(metrics) > 1:
                    # Multiple metrics over time - use multi-series line chart
                    config["options"]["interaction"] = {
                        "mode": "index",
                        "intersect": False,
                    }

                # Configure time scale
                config["options"]["scales"] = {
                    "x": {
                        "type": "time",
                        "time": {
                            "unit": "day",  # Can be made dynamic based on data range
                            "displayFormats": {"day": "MMM D, YYYY"},
                        },
                        "title": {
                            "display": True,
                            "text": temporal_dim.replace("METRIC_TIME_", "")
                            .replace("_", " ")
                            .title(),
                        },
                    },
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": ", ".join(metrics)},
                    },
                }

                # Update datasets with timestamps
                config["data"]["labels"] = timestamps
                config["data"]["datasets"] = [
                    {
                        "label": metric,
                        "data": [
                            {"x": timestamps[i], "y": data.get(metric.upper(), [])[i]}
                            for i in range(len(timestamps))
                        ],
                        "borderColor": f"hsl({hash(metric) % 360}, 70%, 50%)",
                        "backgroundColor": f"hsl({hash(metric) % 360}, 70%, 50%, 0.5)",
                        "tension": 0.3,  # Add slight curve to lines
                    }
                    for metric in metrics
                ]

            # Case 3: Single categorical dimension
            elif len(group_by) == 1:
                if unique_values <= 7:  # Based on article recommendation
                    config["type"] = "bar"
                    if len(metrics) > 1:
                        # Use grouped bar chart for multiple metrics
                        config["data"]["datasets"] = [
                            {
                                "label": metric,
                                "data": data.get(metric.upper(), []),
                                "backgroundColor": f"hsl({hash(metric) % 360}, 70%, 50%)",
                            }
                            for metric in metrics
                        ]
                else:
                    # Many categories - use horizontal bar
                    config["type"] = "bar"
                    config["options"]["indexAxis"] = "y"  # This makes it horizontal

            # Case 4: Multiple dimensions
            else:
                if len(metrics) == 1:
                    # Single metric with multiple dimensions - use stacked bar
                    config["type"] = "bar"
                    config["options"]["scales"] = {
                        "x": {"stacked": True},
                        "y": {"stacked": True},
                    }
                else:
                    # Multiple metrics and dimensions - use grouped bar
                    config["type"] = "bar"
                    if unique_values > 7:
                        config["options"]["indexAxis"] = "y"  # This makes it horizontal

            # Set common data properties
            config["data"]["labels"] = [str(v) for v in primary_values]
            if not config["data"].get("datasets"):
                config["data"]["datasets"] = [
                    {
                        "label": metric,
                        "data": data.get(metric.upper(), []),
                        "borderColor": f"hsl({hash(metric) % 360}, 70%, 50%)",
                        "backgroundColor": f"hsl({hash(metric) % 360}, 70%, 50%, 0.5)",
                    }
                    for metric in metrics
                ]

            return config
        except Exception as e:
            logger.error(f"Error determining chart type: {e}")
            # Return a minimal valid chart config on error
            return {
                "type": "bar",
                "data": {"labels": [], "datasets": []},
                "options": {"responsive": True},
            }

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
            # Ensure defaults for all fields
            group_by = group_by or []
            order_by = order_by or []
            where = where or []

            logger.debug(
                f"Querying semantic layer with metrics: {metrics}, "
                f"group_by: {group_by}, limit: {limit}, "
                f"order_by: {order_by}, where: {where}"
            )

            # Execute query and get SQL concurrently
            table, sql = await asyncio.gather(
                self.app.state.client.query(
                    metrics=metrics,
                    group_by=group_by,
                    limit=limit,
                    order_by=order_by,
                    where=where,
                ),
                self.app.state.client.compile_sql(
                    metrics=metrics,
                    group_by=group_by,
                    limit=limit,
                    order_by=order_by,
                    where=where,
                ),
            )

            logger.debug("Query completed successfully")

            # Convert table to dict and format the data
            data_dict = table.to_pydict()
            formatted_data = self._format_data(data_dict)

            # Generate chart configuration using our internal logic
            chart_config = self._determine_chart_type(
                data=formatted_data, metrics=metrics, group_by=group_by or []
            )

            # Format the response for the frontend - ensure it's wrapped correctly for direct return
            return {
                "type": "function_call_output",  # This matches what the frontend expects
                "output": json.dumps(
                    {
                        "type": "query_result",
                        "sql": sql,
                        "data": formatted_data,
                        "chart_config": chart_config,
                        "metrics": metrics,
                    }
                ),
            }

        except Exception as e:
            logger.error(f"Error in semantic layer query: {e}")
            return {"error": str(e), "type": "error"}


def create_tools(app: Starlette) -> Sequence[BaseTool]:
    """Create the tools with access to application state."""
    tavily_tool = TavilySearchResults(
        max_results=5,
        include_answer=True,
        description=(
            "This is a search tool for accessing the internet.\n\n"
            "Let the user know you're asking your friend Tavily for help before you call the tool."
        ),
    )

    return [
        SemanticLayerSearchTool(app=app),
        SemanticLayerQueryTool(app=app),
        tavily_tool,
    ]
