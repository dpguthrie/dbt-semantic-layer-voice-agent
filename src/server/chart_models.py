from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

# Re-using color constants from original implementation
COLORS = [
    (94, 234, 212),
    (94, 186, 234),
    (94, 116, 234),
    (142, 94, 234),
    (212, 94, 234),
    (234, 94, 186),
]

CHART_STANDARD_OPTIONS = {
    "responsive": True,
    "plugins": {
        "interaction": {
            "intersect": False,
            "mode": "nearest",
        },
        "legend": {
            "position": "bottom",
        },
    },
}


class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    DOUGHNUT = "doughnut"
    STACKED_BAR = "stacked_bar"


class ColorStrategy(Enum):
    SINGLE = "single"
    SEPARATE = "separate"


def get_rgb(idx: int) -> tuple[int, int, int]:
    try:
        return COLORS[idx % len(COLORS)]
    except IndexError:
        return (
            np.random.randint(50, 150),
            np.random.randint(75, 255),
            np.random.randint(75, 255),
        )


class BaseChart(BaseModel, ABC):
    """Abstract base class for all chart types"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: list[str]
    dimensions: list[str]
    table: pa.Table

    @property
    def has_time_dimension(self) -> bool:
        return any(pa.types.is_temporal(field.type) for field in self.table.schema)

    @property
    def time_dimension(self) -> str | None:
        for field in self.table.schema:
            if pa.types.is_temporal(field.type):
                return field.name
        return None

    def number_of_unique_values_for(self, dimension: str) -> int:
        return len(self.table.column(dimension).unique())

    @abstractmethod
    def prepare_data(self) -> pd.DataFrame:
        """Transform the data into the format needed for this chart type"""
        pass

    @abstractmethod
    def create_datasets(self, df: pd.DataFrame, x_axis_label: str) -> list[dict]:
        """Create the datasets specific to this chart type"""
        pass

    @abstractmethod
    def apply_colors(self, datasets: list[dict]) -> None:
        """Apply color strategy specific to this chart type"""
        pass

    @property
    def chart_type(self) -> ChartType:
        """Return the chart type for this class"""
        pass

    @property
    def color_strategy(self) -> ColorStrategy:
        """Return the color strategy for this class"""
        pass

    def get_labels(self) -> tuple[str | None, list[Any]]:
        """Get x-axis labels and values"""
        if self.has_time_dimension:
            time_values = self.table.column(self.time_dimension).to_pylist()
            time_ms = sorted(
                [int(pd.Timestamp(dt).timestamp() * 1000) for dt in time_values]
            )
            return self.time_dimension, time_ms

        if len(self.metrics) == len(self.table.schema):
            return None, self.metrics

        if len(self.dimensions) == 1:
            dim = self.dimensions[0]
            return dim, self.table.column(dim).to_pylist()

        # Find dimension with least unique values for x-axis
        dims = {dim: self.number_of_unique_values_for(dim) for dim in self.dimensions}
        least_unique_dim = min(dims.keys(), key=dims.get)
        return least_unique_dim, self.table.column(least_unique_dim).to_pylist()

    def get_config(self) -> dict:
        """Generate chart configuration"""
        x_axis_label, x_axis_values = self.get_labels()
        df = self.prepare_data()
        datasets = self.create_datasets(df, x_axis_label)
        self.apply_colors(datasets)

        options = CHART_STANDARD_OPTIONS.copy()
        if self.has_time_dimension:
            options["scales"] = {"x": {"type": "timeseries"}}

        return {
            "type": self.chart_type.value,
            "data": {
                "labels": x_axis_values,
                "datasets": datasets,
            },
            "options": options,
        }


class TimeSeriesChart(BaseChart):
    """Chart specialized for time series data"""

    def prepare_data(self) -> pd.DataFrame:
        df = self.table.to_pandas()

        # Sort by time dimension in ascending order
        df = df.sort_values(by=self.time_dimension)

        # Convert to milliseconds for JavaScript
        df[self.time_dimension] = (
            pd.to_datetime(df[self.time_dimension]).astype(np.int64) / 1e6
        )
        df[self.metrics] = df[self.metrics].astype(np.float64)

        # Use wide format if we have multiple dimensions
        if len(self.dimensions) > 1:
            index = self.time_dimension
            columns = [d for d in self.dimensions if d != index]
            df = df.pivot(index=index, columns=columns, values=self.metrics)

        return df

    def create_datasets(self, df: pd.DataFrame, x_axis_label: str) -> list[dict]:
        datasets = []
        for series_name, series in df.items():
            if series_name != x_axis_label:
                label = (
                    " - ".join(
                        str(part).replace("_", " ").title() for part in series_name
                    )
                    if isinstance(series_name, tuple)
                    else str(series_name).replace("_", " ").title()
                )
                datasets.append(
                    {
                        "label": label,
                        "data": series.tolist(),
                    }
                )
        return datasets

    def apply_colors(self, datasets: list[dict]) -> None:
        for i, dataset in enumerate(datasets):
            r, g, b = get_rgb(i)
            dataset["backgroundColor"] = f"rgb({r}, {g}, {b})"
            dataset["borderColor"] = f"rgb({r}, {g}, {b})"

    @property
    def chart_type(self) -> ChartType:
        unique_values = self.number_of_unique_values_for(self.time_dimension)
        return ChartType.LINE if unique_values > 5 else ChartType.BAR

    @property
    def color_strategy(self) -> ColorStrategy:
        return (
            ColorStrategy.SEPARATE
            if len(self.metrics) > 1 or len(self.dimensions) > 1
            else ColorStrategy.SINGLE
        )


class DoughnutChart(BaseChart):
    """Specialized chart for categorical data with few unique values"""

    def prepare_data(self) -> pd.DataFrame:
        df = self.table.to_pandas()
        df[self.metrics] = df[self.metrics].astype(np.float64)
        return df

    def create_datasets(self, df: pd.DataFrame, x_axis_label: str) -> list[dict]:
        return [
            {
                "label": metric.replace("_", " ").title(),
                "data": df[metric].tolist(),
            }
            for metric in self.metrics
        ]

    def apply_colors(self, datasets: list[dict]) -> None:
        for dataset in datasets:
            colors = [
                f"rgb({r}, {g}, {b})"
                for r, g, b in (get_rgb(i) for i in range(len(dataset["data"])))
            ]
            dataset["backgroundColor"] = colors

    @property
    def chart_type(self) -> ChartType:
        return ChartType.DOUGHNUT

    @property
    def color_strategy(self) -> ColorStrategy:
        return ColorStrategy.SEPARATE


class StackedBarChart(BaseChart):
    """Specialized chart for multi-dimensional categorical data"""

    def prepare_data(self) -> pd.DataFrame:
        df = self.table.to_pandas()
        df[self.metrics] = df[self.metrics].astype(np.float64)
        x_axis_label, _ = self.get_labels()
        columns = [d for d in self.dimensions if d != x_axis_label]
        return df.pivot(index=x_axis_label, columns=columns, values=self.metrics)

    def create_datasets(self, df: pd.DataFrame, x_axis_label: str) -> list[dict]:
        datasets = []
        for series_name, series in df.items():
            if series_name != x_axis_label:
                label = (
                    " - ".join(
                        str(part).replace("_", " ").title() for part in series_name
                    )
                    if isinstance(series_name, tuple)
                    else str(series_name).replace("_", " ").title()
                )
                datasets.append(
                    {
                        "label": label,
                        "data": series.tolist(),
                    }
                )
        return datasets

    def apply_colors(self, datasets: list[dict]) -> None:
        for i, dataset in enumerate(datasets):
            r, g, b = get_rgb(i)
            dataset["backgroundColor"] = f"rgb({r}, {g}, {b})"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.STACKED_BAR

    @property
    def color_strategy(self) -> ColorStrategy:
        return ColorStrategy.SEPARATE


class BarChart(BaseChart):
    """Standard bar chart for categorical data"""

    def prepare_data(self) -> pd.DataFrame:
        df = self.table.to_pandas()
        df[self.metrics] = df[self.metrics].astype(np.float64)
        # If table contains all metrics, keep as is
        if len(self.metrics) == len(self.table.schema):
            return df

        if len(self.dimensions) > 1:
            x_axis_label, _ = self.get_labels()
            columns = [d for d in self.dimensions if d != x_axis_label]
            df = df.pivot(index=x_axis_label, columns=columns, values=self.metrics)

        return df

    def create_datasets(self, df: pd.DataFrame, x_axis_label: str) -> list[dict]:
        # Special handling for tables containing only metrics
        if len(self.metrics) == len(self.table.schema):
            return [
                {
                    "label": "Values",
                    "data": [df[metric].iloc[0] for metric in self.metrics],
                }
            ]

        # Regular handling for tables with dimensions
        datasets = []
        for series_name, series in df.items():
            if series_name != x_axis_label:
                label = (
                    " - ".join(
                        str(part).replace("_", " ").title() for part in series_name
                    )
                    if isinstance(series_name, tuple)
                    else str(series_name).replace("_", " ").title()
                )
                datasets.append(
                    {
                        "label": label,
                        "data": series.tolist(),
                    }
                )
        return datasets

    def apply_colors(self, datasets: list[dict]) -> None:
        # For tables with only metrics, always use separate colors
        if len(self.metrics) == len(self.table.schema):
            values = len(self.metrics)
            colors = [
                f"rgb({r}, {g}, {b})" for r, g, b in (get_rgb(i) for i in range(values))
            ]
            datasets[0]["backgroundColor"] = colors
            datasets[0]["borderColor"] = colors
            datasets[0]["borderWidth"] = 1
            return

        # Regular color handling
        if len(datasets) == 1 and self.color_strategy == ColorStrategy.SEPARATE:
            # For single dataset with separate colors, each bar gets its own color
            values = len(datasets[0]["data"])
            colors = [
                f"rgb({r}, {g}, {b})" for r, g, b in (get_rgb(i) for i in range(values))
            ]
            datasets[0]["backgroundColor"] = colors
        else:
            # For multiple datasets or single color strategy, each dataset gets one color
            for i, dataset in enumerate(datasets):
                r, g, b = get_rgb(i)
                dataset["backgroundColor"] = f"rgb({r}, {g}, {b})"

    @property
    def chart_type(self) -> ChartType:
        return ChartType.BAR

    @property
    def color_strategy(self) -> ColorStrategy:
        # Always use separate colors for metrics-only tables
        if len(self.metrics) == len(self.table.schema):
            return ColorStrategy.SEPARATE

        # Use separate colors if we have a single metric and multiple dimensions
        # or if we have a single dimension with a single metric
        return (
            ColorStrategy.SEPARATE
            if (len(self.metrics) == 1 and len(self.dimensions) > 1)
            or (len(self.metrics) == 1 and len(self.dimensions) == 1)
            else ColorStrategy.SINGLE
        )


def create_chart(
    metrics: list[str], dimensions: list[str], table: pa.Table
) -> BaseChart:
    """Factory function to create the appropriate chart type based on data characteristics"""

    def convert_to_uppercase(v: list[str]) -> list[str]:
        return [s.upper() for s in v]

    metrics = convert_to_uppercase(metrics)
    dimensions = convert_to_uppercase(dimensions)

    # Check if time series
    has_time = any(pa.types.is_temporal(field.type) for field in table.schema)
    if has_time:
        return TimeSeriesChart(metrics=metrics, dimensions=dimensions, table=table)

    # Check if table contains all metrics (no dimensions)
    if len(metrics) == len(table.schema):
        return BarChart(metrics=metrics, dimensions=dimensions, table=table)

    # Check if suitable for doughnut chart (2 columns, few unique values)
    if len(table.schema) == 2:
        first_dim = dimensions[0]
        if len(table.column(first_dim).unique()) <= 5:
            return DoughnutChart(metrics=metrics, dimensions=dimensions, table=table)

    # Use stacked bar for 3+ dimensions
    if len(dimensions) >= 3:
        return StackedBarChart(metrics=metrics, dimensions=dimensions, table=table)

    # Default to standard bar chart for all other cases
    return BarChart(metrics=metrics, dimensions=dimensions, table=table)
