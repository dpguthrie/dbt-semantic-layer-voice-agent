import braintrust
from server.client import get_client
from server.settings import settings
from server.tools import SemanticLayerQueryTool, SemanticLayerSearchTool
from server.vectorstore import SemanticLayerVectorStore

project = braintrust.projects.get(settings.braintrust_project_name)
semantic_layer_client = get_client()
vector_store = SemanticLayerVectorStore(semantic_layer_client)


def create_tools_on_braintrust():
    tools = [
        SemanticLayerSearchTool(vector_store=vector_store),
        SemanticLayerQueryTool(semantic_layer_client),
    ]

    for tool in tools:
        project.tools.create(
            handler=tool._arun,
            name=tool.name,
            description=tool.description,
            parameters=tool.args_schema.model_json_schema(),
            returns=tool.return_type,
        )
