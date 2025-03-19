from server.tools import SemanticLayerMetadataTool, SemanticLayerQueryTool


async def semantic_layer_pipeline(query: str):
    metadata_tool = SemanticLayerMetadataTool()
    query_tool = SemanticLayerQueryTool()

    metadata = await metadata_tool.arun(query)
    query = await query_tool.arun(metadata)

    return query
