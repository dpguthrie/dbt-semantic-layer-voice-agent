from datetime import datetime
from typing import Any

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


class QueryParameters(BaseModel):
    """The parameters of a query to the semantic layer."""

    metrics: list[str]
    group_by: list[str] = Field(default_factory=list)
    limit: int | None = None
    order_by: list[str] = Field(
        default_factory=list
    )  # Format: "-metric_name" for desc, "metric_name" for asc
    where: list[str] = Field(default_factory=list)


class Message(BaseModel):
    """A message in a conversation."""

    id: int | None = None
    text: str
    is_user: bool
    timestamp: datetime
    data: dict[str, Any] | None = None  # For storing JSON responses with charts/tables
    conversation_id: int  # ID of the conversation this message belongs to


class Conversation(BaseModel):
    """A conversation with messages."""

    id: int
    title: str
    messages: list[Message]
    context: str | None = Field(
        None,
        description="Optional text context that applies to all messages in the conversation (e.g., date ranges, filters)",
    )
    created_at: datetime
    updated_at: datetime
