"""
Global settings for the server application using Pydantic Settings.
Settings can be overridden using environment variables.
For example, SL__HOST will set the host field in SemanticLayerSettings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SemanticLayerSettings(BaseSettings):
    """Semantic layer settings."""

    host: str
    environment_id: int
    token: str


class Settings(BaseSettings):
    """Global application settings that can be overridden by environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    sl: SemanticLayerSettings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-realtime-preview-2024-12-17"
    braintrust_api_key: str = ""
    braintrust_project_name: str = ""
    pinecone_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_metric_index_name: str = "semantic-metrics"
    pinecone_dimension_index_name: str = "semantic-dimensions"


settings = Settings()
