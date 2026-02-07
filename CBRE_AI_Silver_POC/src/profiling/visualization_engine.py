"""Visualization engine for data profiling results.

Uses Altair (bundled with Streamlit) for all chart generation.
Charts can be rendered in Streamlit or exported as SVG for HTML reports.
"""

import altair as alt
import pandas as pd
from typing import Dict, List, Any


def build_completeness_chart(profiles: Dict[str, Dict[str, Any]]) -> alt.Chart:
    """Horizontal bar chart of column completeness, color-coded by quality tier."""
    rows = []
    for table, cols in profiles.items():
        for col_name, stats in cols.items():
            total = stats.get("total", 0)
            nulls = stats.get("nulls", 0)
            completeness = round((total - nulls) / total * 100, 1) if total else 0
            rows.append({
                "Column": f"{table}.{col_name}" if len(profiles) > 1 else col_name,
                "Completeness": completeness,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data")

    df = df.sort_values("Completeness", ascending=True)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Completeness:Q", scale=alt.Scale(domain=[0, 100]), title="Completeness %"),
            y=alt.Y("Column:N", sort=None, title=""),
            color=alt.condition(
                alt.datum.Completeness > 90,
                alt.value("#2ecc71"),
                alt.condition(
                    alt.datum.Completeness > 50,
                    alt.value("#f39c12"),
                    alt.value("#e74c3c"),
                ),
            ),
            tooltip=["Column", "Completeness"],
        )
        .properties(title="Column Completeness", height=max(200, len(rows) * 22))
    )
    return chart


def build_datatype_distribution_chart(profiles: Dict[str, Dict[str, Any]]) -> alt.Chart:
    """Donut chart showing distribution of data types across columns."""
    type_counts = {}
    for table, cols in profiles.items():
        for col_name, stats in cols.items():
            dtype = stats.get("type_category", stats.get("datatype", "other"))
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

    df = pd.DataFrame([{"Type": k, "Count": v} for k, v in type_counts.items()])
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data")

    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Type:N", scale=alt.Scale(scheme="category10")),
            tooltip=["Type", "Count"],
        )
        .properties(title="Data Type Distribution", width=300, height=300)
    )
    return chart


def build_null_heatmap(profiles: Dict[str, Dict[str, Any]]) -> alt.Chart:
    """Heatmap of null percentages across tables and columns."""
    rows = []
    for table, cols in profiles.items():
        for col_name, stats in cols.items():
            total = stats.get("total", 0)
            nulls = stats.get("nulls", 0)
            null_pct = round(nulls / total * 100, 1) if total else 0
            rows.append({
                "Table": table,
                "Column": col_name,
                "Null %": null_pct,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data")

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Column:N", title=""),
            y=alt.Y("Table:N", title=""),
            color=alt.Color(
                "Null %:Q",
                scale=alt.Scale(scheme="orangered", domain=[0, 100]),
                title="Null %",
            ),
            tooltip=["Table", "Column", "Null %"],
        )
        .properties(title="Null Percentage Heatmap", height=max(150, len(profiles) * 40))
    )
    return chart


def build_string_length_histogram(sample_values: List[str], col_name: str) -> alt.Chart:
    """Histogram of string lengths from sample values."""
    if not sample_values:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No sample data")

    lengths = [len(str(v)) for v in sample_values if v]
    df = pd.DataFrame({"Length": lengths})

    chart = (
        alt.Chart(df)
        .mark_bar(color="#3498db")
        .encode(
            x=alt.X("Length:Q", bin=alt.Bin(maxbins=20), title="String Length"),
            y=alt.Y("count()", title="Frequency"),
            tooltip=["count()"],
        )
        .properties(title=f"String Length Distribution: {col_name}", width=400, height=250)
    )
    return chart


def build_value_frequency_chart(top_values: List[Dict], col_name: str) -> alt.Chart:
    """Bar chart of top value frequencies."""
    if not top_values:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data")

    df = pd.DataFrame(top_values)
    if "value" not in df.columns or "count" not in df.columns:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data")

    # Truncate long value labels
    df["label"] = df["value"].apply(lambda x: x[:30] + "..." if len(str(x)) > 30 else str(x))

    chart = (
        alt.Chart(df)
        .mark_bar(color="#9b59b6")
        .encode(
            x=alt.X("count:Q", title="Frequency"),
            y=alt.Y("label:N", sort="-x", title=""),
            tooltip=["value", "count"],
        )
        .properties(title=f"Top Values: {col_name}", width=400, height=max(150, len(df) * 30))
    )
    return chart


def build_quality_score_gauge(deep_profiles: Dict[str, Dict[str, Any]]) -> alt.Chart:
    """Quality score bar chart across all columns from deep profiler results."""
    rows = []
    for col_name, dp in deep_profiles.items():
        score = dp.get("data_quality_score")
        if score is not None:
            rows.append({"Column": col_name, "Quality Score": score})

    if not rows:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No quality scores available")

    df = pd.DataFrame(rows).sort_values("Quality Score", ascending=True)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Quality Score:Q", scale=alt.Scale(domain=[0, 100]), title="Quality Score"),
            y=alt.Y("Column:N", sort=None, title=""),
            color=alt.condition(
                alt.datum["Quality Score"] > 80,
                alt.value("#2ecc71"),
                alt.condition(
                    alt.datum["Quality Score"] > 50,
                    alt.value("#f39c12"),
                    alt.value("#e74c3c"),
                ),
            ),
            tooltip=["Column", "Quality Score"],
        )
        .properties(title="AI Quality Scores", height=max(200, len(rows) * 22))
    )
    return chart
