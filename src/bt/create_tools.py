import asyncio

import braintrust
from server.settings import settings
from server.tools import SemanticLayerQueryTool, SemanticLayerSearchTool


async def create_tools_on_braintrust():
    # Create or get existing project
    try:
        project = braintrust.projects.create(settings.braintrust_project_name)
        print(f"Project '{settings.braintrust_project_name}' initialized")
    except Exception as e:
        print(f"Error creating/accessing project: {e}")
        raise

    tools = [
        SemanticLayerSearchTool(),
        SemanticLayerQueryTool(),
    ]

    for tool in tools:
        project.tools.create(
            handler=tool._arun,
            name=tool.name,
            description=tool.description,
            parameters=tool.args_schema,
        )
        print(f"Created tool: {tool.name}")


asyncio.run(create_tools_on_braintrust())
