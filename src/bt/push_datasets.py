# stdlib
import os

# third party
import braintrust
from dotenv import load_dotenv

load_dotenv()

BRAINTRUST_PROJECT_NAME = os.getenv("BRAINTRUST_PROJECT_NAME")
BRAINTRUST_QUERY_DATASET_NAME = "semantic_layer_query_examples"
BRAINTRUST_METADATA_DATASET_NAME = "semantic_layer_metadata_examples"

QUERY_EXAMPLES = [
    {
        "input": "Show me total revenue",
        "expected": {
            "metrics": ["total_revenue"],
            "group_by": [],
            "where": [],
            "order_by": [],
            "limit": None,
        },
    },
    {
        "input": "What is monthly revenue and profit for 2023",
        "expected": {
            "metrics": ["total_revenue", "total_profit"],
            "group_by": ["metric_time__month"],
            "where": [
                "{{ TimeDimension('metric_time', 'DAY') }} between '2023-01-01' and '2023-12-31'"
            ],
            "order_by": ["metric_time__month"],
            "limit": None,
        },
    },
    {
        "input": "What's revenue by region",
        "expected": {
            "metrics": ["total_revenue"],
            "group_by": ["customer__region"],
            "order_by": [],
            "limit": None,
            "where": [],
        },
    },
    {
        "input": "Show me daily revenue for the past 30 days",
        "expected": {
            "metrics": ["total_revenue"],
            "group_by": ["metric_time__day"],
            "where": [
                "{{ TimeDimension('metric_time', 'DAY') }} >= dateadd('day', -30, current_date)"
            ],
            "order_by": ["metric_time__day"],
            "limit": None,
        },
    },
    {
        "input": "What is the weekly customer count this quarter",
        "expected": {
            "metrics": ["weekly_customers"],
            "group_by": ["metric_time__week"],
            "where": [
                "{{ TimeDimension('metric_time', 'DAY') }} >= date_trunc('quarter', current_date)"
            ],
            "order_by": ["metric_time__week"],
            "limit": None,
        },
    },
    {
        "input": "What are the top 10 customers by total revenue",
        "expected": {
            "metrics": ["total_revenue"],
            "group_by": ["customer__customer_name"],
            "where": [],
            "order_by": ["-total_revenue"],
            "limit": 10,
        },
    },
    {
        "input": "Filter all results to Q2-2023 and show me monthly customers",
        "expected": {
            "metrics": ["monthly_customers"],
            "group_by": ["metric_time__month"],
            "where": [
                "{{ TimeDimension('metric_time', 'DAY') }} between '2023-04-01' and '2023-06-30'"
            ],
            "order_by": ["metric_time__month"],
            "limit": None,
        },
    },
    {
        "input": "Order by metric_time ascending and show me cumulative revenue by region",
        "expected": {
            "metrics": ["cumulative_revenue_total"],
            "group_by": ["customer__region", "metric_time__day"],
            "order_by": ["metric_time__day"],
            "limit": None,
            "where": [],
        },
    },
    {
        "input": "Only include HIGH balance segment and show me our average revenue per customer?",
        "expected": {
            "metrics": ["average_revenue_per_customer"],
            "where": [
                "{{ Dimension('customer__customer_balance_segment') }} ilike 'HIGH'"
            ],
            "limit": None,
            "group_by": [],
            "order_by": [],
        },
    },
    {
        "input": "Include both revenue and customer count by market segment",
        "expected": {
            "metrics": ["total_revenue", "total_customers"],
            "group_by": ["customer__customer_market_segment"],
            "limit": None,
            "where": [],
            "order_by": [],
        },
    },
    {
        "input": "TTM revenue trend by region",
        "expected": {
            "metrics": ["cumulative_revenue_ttm"],
            "group_by": ["customer__region", "metric_time__month"],
            "order_by": ["metric_time__month"],
            "limit": None,
            "where": [],
        },
    },
    {
        "input": "Year to date profit by nation",
        "expected": {
            "metrics": ["total_profit"],
            "group_by": ["customer__nation"],
            "where": [
                "{{ TimeDimension('metric_time', 'DAY') }} >= date_trunc('year', current_date)"
            ],
            "order_by": ["-total_profit"],
            "limit": None,
        },
    },
]

METADATA_EXAMPLES = [
    {
        "input": "What metrics can I query?",
        "expected": """Here are the available metrics you can query:

* Revenue Metrics:
  - `total_revenue`: Total revenue across all customers
  - `average_revenue_per_customer`: Average revenue per customer
  - `cumulative_revenue_total`: Running total of revenue
  - `cumulative_revenue_ttm`: Trailing twelve months revenue

* Customer Metrics:
  - `total_customers`: Total count of unique customers
  - `monthly_customers`: Count of customers per month
  - `weekly_customers`: Count of customers per week

* Profitability:
  - `total_profit`: Total profit across all transactions""",
    },
    {
        "input": "What dimensions are available for analyzing revenue?",
        "expected": """You can analyze revenue across these dimensions:

* Time-based Dimensions:
  - `metric_time__day`: Daily breakdown
  - `metric_time__week`: Weekly aggregation
  - `metric_time__month`: Monthly view
  - `metric_time__quarter`: Quarterly analysis
  - `metric_time__year`: Yearly totals

* Customer Attributes:
  - `customer__region`: Geographic region
  - `customer__nation`: Country/nation
  - `customer__customer_name`: Individual customer
  - `customer__customer_market_segment`: Market segment classification
  - `customer__customer_balance_segment`: Balance tier (HIGH/MEDIUM/LOW)""",
    },
    {
        "input": "Tell me about customer segmentation metrics",
        "expected": """Here are the metrics and dimensions related to customer segmentation:

* Key Metrics:
  - `monthly_customers`: Track customer count monthly
  - `average_revenue_per_customer`: Understand customer value
  - `total_customers`: Overall customer base size

* Available Segmentation Dimensions:
  - `customer__customer_market_segment`: Segment by market type
  - `customer__customer_balance_segment`: Group by account balance
  - `customer__region`: Regional segmentation

You can combine these metrics and dimensions to analyze customer behavior across different segments.""",
    },
    {
        "input": "What time-based metrics are available?",
        "expected": """Here are the time-based metrics and analysis options available:

* Periodic Customer Metrics:
  - `weekly_customers`: Weekly active customer count
  - `monthly_customers`: Monthly active customer count

* Cumulative Metrics:
  - `cumulative_revenue_total`: Progressive revenue total over time
  - `cumulative_revenue_ttm`: Trailing twelve months revenue analysis

* Time Granularity Options:
  - `metric_time__day`: Daily analysis
  - `metric_time__week`: Weekly rollup
  - `metric_time__month`: Monthly aggregation
  - `metric_time__quarter`: Quarterly view

These can be combined with other metrics to analyze trends and patterns over time.""",
    },
    {
        "input": "How can I analyze profitability?",
        "expected": """Here are the metrics and dimensions available for profitability analysis:

* Core Profitability Metrics:
  - `total_profit`: Overall profit across all transactions
  - `total_revenue`: Gross revenue before costs

* Related Analysis Dimensions:
  - `customer__region`: Analyze profitability by geographic region
  - `customer__market_segment`: Understand profit across market segments
  - `customer__nation`: Compare profitability across countries

* Time-based Analysis Options:
  - Use `metric_time__month` for monthly profit trends
  - Use `metric_time__quarter` for quarterly performance
  - Use `metric_time__year` for annual profitability review""",
    },
    {
        "input": "What geographic analysis options do I have?",
        "expected": """You can analyze data across these geographic dimensions:

* Geographic Hierarchy:
  - `customer__region`: Broad regional analysis
  - `customer__nation`: Country-level breakdown

* Available Metrics for Geographic Analysis:
  - `total_revenue`: Revenue by location
  - `total_profit`: Profitability by area
  - `total_customers`: Customer distribution
  - `average_revenue_per_customer`: Customer value by region

* Common Geographic Analysis Patterns:
  - Combine with `metric_time__month` for regional trends
  - Use with `customer__market_segment` for geo-market analysis
  - Pair with cumulative metrics for regional growth tracking""",
    },
    {
        "input": "What metrics help me understand customer value?",
        "expected": """Here are the metrics and dimensions for analyzing customer value:

* Direct Value Metrics:
  - `average_revenue_per_customer`: Individual customer worth
  - `total_revenue`: Aggregate customer value
  - `total_profit`: Customer profitability

* Customer Engagement Metrics:
  - `weekly_customers`: Short-term engagement tracking
  - `monthly_customers`: Medium-term engagement analysis

* Value Analysis Dimensions:
  - `customer__customer_balance_segment`: Value tiers (HIGH/MEDIUM/LOW)
  - `customer__customer_market_segment`: Value by market type
  - `customer__customer_name`: Individual customer analysis

These can be combined to create comprehensive customer value profiles and segmentation analysis.""",
    },
    {
        "input": "What are the cumulative metrics I can use?",
        "expected": """Here are the available cumulative metrics and their analysis options:

* Revenue-based Cumulative Metrics:
  - `cumulative_revenue_total`: Running total of revenue over time
  - `cumulative_revenue_ttm`: Trailing twelve months revenue

* Time Dimensions for Cumulative Analysis:
  - `metric_time__day`: Daily cumulative totals
  - `metric_time__month`: Monthly accumulation
  - `metric_time__quarter`: Quarterly running totals

* Common Analysis Patterns:
  - Combine with `customer__region` for regional growth tracking
  - Use with market segments for cumulative segment performance
  - Pair with customer dimensions for cohort analysis

These metrics are particularly useful for tracking growth trends and performing year-over-year comparisons.""",
    },
]


query_dataset = braintrust.init_dataset(
    project=BRAINTRUST_PROJECT_NAME, name=BRAINTRUST_QUERY_DATASET_NAME
)

for example in QUERY_EXAMPLES:
    id = query_dataset.insert(
        input=example["input"],
        expected=example["expected"],
    )
    print(f"Inserted record with id: {id}")

metadata_dataset = braintrust.init_dataset(
    project=BRAINTRUST_PROJECT_NAME, name=BRAINTRUST_METADATA_DATASET_NAME
)

for example in METADATA_EXAMPLES:
    id = metadata_dataset.insert(
        input=example["input"],
        expected=example["expected"],
    )
    print(f"Inserted record with id: {id}")
