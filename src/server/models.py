from pydantic import BaseModel, Field, field_validator


class _PageContentModel(BaseModel):
    """
    Base class for models that have a page_content property.
    """

    @property
    def page_content(self) -> str:
        page_content = self.name
        if self.label:
            page_content += f" - {self.label}"
        if self.description:
            page_content += f": {self.description}"
        return page_content


class VectorStoreMetric(_PageContentModel):
    """
    Metric retrieved from the user's semantic layer config.
    """

    name: str
    metric_type: str
    requires_metric_time: bool
    dimensions: str
    queryable_granularities: str
    label: str | None = ""
    description: str | None = ""

    @field_validator("label", "description", mode="before")
    def empty_string_for_none(cls, v):
        # Convert None to empty string to ensure these fields are never None
        return "" if v is None else v


class VectorStoreDimension(_PageContentModel):
    """
    Dimension retrieved from the user's semantic layer config.
    """

    name: str
    dimension_type: str
    qualified_name: str
    metric_id: str
    label: str | None = ""
    description: str | None = ""
    expr: str | None = ""

    @field_validator("label", "description", "expr", mode="before")
    def empty_string_for_none(cls, v):
        # Convert None to empty string to ensure these fields are never None
        return "" if v is None else v


class RetrievalMetric(BaseModel):
    """Simplified metric model for retrieval results."""

    name: str = Field(description="Unique identifier of the metric")
    label: str = Field(description="Human-readable label of the metric")
    description: str | None = Field(
        None, description="Optional description of the metric"
    )
    metric_type: str = Field(
        description="Type of the metric (e.g., 'count', 'sum', etc.)"
    )
    requires_metric_time: bool = Field(
        description="Whether the metric requires a metric time"
    )


class RetrievalDimension(BaseModel):
    """Simplified dimension model for retrieval results."""

    name: str = Field(description="Name of the dimension")
    label: str | None = Field(None, description="Human-readable label of the dimension")
    description: str | None = Field(
        None, description="Optional description of the dimension"
    )
    metric_id: str = Field(description="ID of the metric this dimension belongs to")


class RetrievalResult(BaseModel):
    """Result from vector store retrieval."""

    metrics: list[RetrievalMetric] = Field(description="Retrieved relevant metrics")
    dimensions: list[RetrievalDimension] = Field(
        description="Retrieved relevant dimensions"
    )
    query: str = Field(description="Original query that was used for retrieval")


class QueryParameters(BaseModel):
    """The parameters of a query to the semantic layer."""

    metrics: list[str]
    group_by: list[str] = Field(default_factory=list)
    limit: int | None = None
    order_by: list[str] = Field(
        default_factory=list
    )  # Format: "-metric_name" for desc, "metric_name" for asc
    where: list[str] = Field(default_factory=list)
