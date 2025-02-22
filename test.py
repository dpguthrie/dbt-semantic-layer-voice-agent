import os

from dbtsl import SemanticLayerClient

from src.server.chart_models import create_chart

client = SemanticLayerClient(
    host="semantic-layer.cloud.getdbt.com",
    environment_id=218762,
    auth_token=os.getenv("DBT_CLOUD_SERVICE_TOKEN"),
)

metrics = ["total_revenue"]
group_by = ["customer__customer_balance_segment"]
where = []

with client.session():
    table = client.query(metrics=metrics, group_by=group_by, where=where)


chart = create_chart(metrics=metrics, dimensions=group_by, table=table)
config = chart.get_config()

print(config)
