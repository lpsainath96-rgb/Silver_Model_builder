"""HTML report generator for profiling and deep analysis results.

Generates a professional, standalone HTML report with inline CSS,
embedding chart SVGs as base64 and presenting all profiling data.
"""

import json
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import altair as alt
    from profiling.visualization_engine import (
        build_completeness_chart,
        build_datatype_distribution_chart,
        build_null_heatmap,
        build_quality_score_gauge,
    )
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False


def _chart_to_svg_base64(chart) -> str:
    """Convert an Altair chart to base64-encoded SVG for HTML embedding."""
    try:
        svg = chart.to_html()
        return svg
    except Exception:
        return "<p>Chart rendering unavailable</p>"


def _severity_badge(severity: str) -> str:
    colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}
    bg = colors.get(severity, "#95a5a6")
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em;">{severity.upper()}</span>'


def _score_color(score) -> str:
    if score is None:
        return "#95a5a6"
    if score >= 80:
        return "#2ecc71"
    if score >= 50:
        return "#f39c12"
    return "#e74c3c"


def generate_html_report(
    profiles: Dict[str, Dict[str, Any]],
    descriptions: Dict[str, str],
    deep_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a complete, professional HTML report with inline CSS."""

    deep_profiles = deep_profiles or {}
    metadata = metadata or {}
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Compute summary stats
    total_columns = sum(len(cols) for cols in profiles.values())
    all_completeness = []
    all_quality_scores = []
    high_issues = 0
    medium_issues = 0
    low_issues = 0

    for table, cols in profiles.items():
        for col_name, stats in cols.items():
            total = stats.get("total", 0)
            nulls = stats.get("nulls", 0)
            comp = round((total - nulls) / total * 100, 1) if total else 0
            all_completeness.append(comp)

            dp = deep_profiles.get(col_name, {})
            score = dp.get("data_quality_score")
            if score is not None:
                all_quality_scores.append(score)
            for issue in dp.get("quality_issues", []):
                sev = issue.get("severity", "low")
                if sev == "high":
                    high_issues += 1
                elif sev == "medium":
                    medium_issues += 1
                else:
                    low_issues += 1

    avg_completeness = round(sum(all_completeness) / len(all_completeness), 1) if all_completeness else 0
    avg_quality = round(sum(all_quality_scores) / len(all_quality_scores), 1) if all_quality_scores else 0

    # Build charts HTML
    charts_html = ""
    if HAS_ALTAIR:
        try:
            completeness_chart = build_completeness_chart(profiles)
            charts_html += f'<div class="chart-container">{_chart_to_svg_base64(completeness_chart)}</div>'
        except Exception:
            pass
        try:
            dtype_chart = build_datatype_distribution_chart(profiles)
            charts_html += f'<div class="chart-container">{_chart_to_svg_base64(dtype_chart)}</div>'
        except Exception:
            pass
        try:
            heatmap = build_null_heatmap(profiles)
            charts_html += f'<div class="chart-container">{_chart_to_svg_base64(heatmap)}</div>'
        except Exception:
            pass
        if deep_profiles:
            try:
                gauge = build_quality_score_gauge(deep_profiles)
                charts_html += f'<div class="chart-container">{_chart_to_svg_base64(gauge)}</div>'
            except Exception:
                pass

    # Build per-column detail cards
    column_cards = ""
    for table, cols in profiles.items():
        for col_name, stats in cols.items():
            total = stats.get("total", 0)
            nulls = stats.get("nulls", 0)
            comp = round((total - nulls) / total * 100, 1) if total else 0
            dtype = stats.get("datatype", "UNKNOWN")
            type_cat = stats.get("type_category", "")
            distinct = stats.get("distinct", 0)

            desc = descriptions.get(col_name, "No description available")
            dp = deep_profiles.get(col_name, {})
            score = dp.get("data_quality_score")
            score_html = f'<span style="color:{_score_color(score)};font-weight:bold;font-size:1.2em;">{score}/100</span>' if score is not None else '<span style="color:#95a5a6;">N/A</span>'

            # String stats
            string_stats_html = ""
            ss = stats.get("string_stats")
            if ss:
                string_stats_html = f"""
                <div class="stat-row">
                    <strong>String Analysis:</strong>
                    Length {ss.get('min_length')}-{ss.get('max_length')} (avg {ss.get('avg_length')}) |
                    {ss.get('upper_case_pct', 0)}% UPPER, {ss.get('lower_case_pct', 0)}% lower |
                    {ss.get('empty_string_count', 0)} empty strings
                </div>"""

            # Patterns
            patterns = stats.get("detected_patterns", [])
            patterns_html = ""
            if patterns:
                pat_items = ", ".join([f"{p['pattern']} ({p['match_pct']}%)" for p in patterns])
                patterns_html = f'<div class="stat-row"><strong>Detected Patterns:</strong> {pat_items}</div>'

            # Quality issues
            issues_html = ""
            for issue in dp.get("quality_issues", []):
                issues_html += f'<div class="issue-item">{_severity_badge(issue.get("severity", "low"))} {issue.get("issue", "")} - {issue.get("description", "")}</div>'

            # Recommendations
            recs = dp.get("silver_recommendations", {})
            recs_html = ""
            if recs and recs.get("recommended_datatype"):
                recs_html = f"""
                <div class="recommendations">
                    <strong>Silver Recommendations:</strong><br>
                    Type: <code>{recs.get('recommended_datatype')}</code> |
                    Name: <code>{recs.get('recommended_name')}</code> |
                    PK Candidate: {'Yes' if recs.get('is_pk_candidate') else 'No'}<br>
                    Transform: <code>{recs.get('transformation_logic', 'Direct copy')}</code>
                </div>"""

            # Business interpretation
            biz = dp.get("business_interpretation", "")
            biz_html = f'<div class="biz-interp"><em>{biz}</em></div>' if biz else ""

            # Sample values
            samples = stats.get("sample_values", [])[:5]
            samples_html = f'<div class="stat-row"><strong>Samples:</strong> {", ".join(str(s) for s in samples)}</div>' if samples else ""

            column_cards += f"""
            <div class="column-card">
                <div class="card-header">
                    <h3>{table}.{col_name}</h3>
                    <div class="score-badge">Quality: {score_html}</div>
                </div>
                <div class="card-meta">
                    <span class="tag">{dtype}</span>
                    <span class="tag">{type_cat}</span>
                    <span>Completeness: <strong>{comp}%</strong></span> |
                    <span>Distinct: <strong>{distinct:,}</strong></span> |
                    <span>Total: <strong>{total:,}</strong></span>
                </div>
                <div class="description">{desc}</div>
                {string_stats_html}
                {patterns_html}
                {samples_html}
                {issues_html}
                {recs_html}
                {biz_html}
            </div>"""

    # Build issues summary sorted by severity
    all_issues = []
    for col_name, dp in deep_profiles.items():
        for issue in dp.get("quality_issues", []):
            all_issues.append({**issue, "column": col_name})

    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_issues.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

    issues_table = ""
    if all_issues:
        issue_rows = ""
        for iss in all_issues:
            issue_rows += f"""<tr>
                <td>{_severity_badge(iss.get('severity', 'low'))}</td>
                <td>{iss.get('column', '')}</td>
                <td>{iss.get('issue', '')}</td>
                <td>{iss.get('description', '')}</td>
            </tr>"""
        issues_table = f"""
        <h2>Data Quality Issues</h2>
        <table class="issues-table">
            <thead><tr><th>Severity</th><th>Column</th><th>Issue</th><th>Description</th></tr></thead>
            <tbody>{issue_rows}</tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Silver Model Builder - Data Profiling Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
    h2 {{ color: #34495e; margin: 30px 0 15px; border-left: 4px solid #3498db; padding-left: 12px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
    .summary-card {{ background: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
    .summary-card .value {{ font-size: 2em; font-weight: bold; }}
    .summary-card .label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
    .chart-container {{ background: #fff; border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .column-card {{ background: #fff; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .card-header h3 {{ color: #2c3e50; font-size: 1.1em; }}
    .card-meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 10px; }}
    .tag {{ background: #ecf0f1; padding: 2px 8px; border-radius: 4px; margin-right: 5px; font-size: 0.85em; }}
    .description {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 0.95em; }}
    .stat-row {{ margin: 8px 0; font-size: 0.9em; }}
    .issue-item {{ margin: 5px 0; padding: 5px 0; border-bottom: 1px solid #ecf0f1; font-size: 0.9em; }}
    .recommendations {{ background: #e8f6e8; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 0.9em; }}
    .biz-interp {{ color: #555; margin: 8px 0; font-size: 0.9em; }}
    .issues-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
    .issues-table th {{ background: #34495e; color: #fff; padding: 10px; text-align: left; }}
    .issues-table td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
    .footer {{ text-align: center; color: #95a5a6; margin: 30px 0; padding: 20px; border-top: 1px solid #ecf0f1; font-size: 0.85em; }}
    code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="container">
    <h1>Silver Model Builder - Data Profiling Report</h1>
    <p style="color:#7f8c8d;">Generated: {generated_at}</p>

    <h2>Executive Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{len(profiles)}</div>
            <div class="label">Tables Profiled</div>
        </div>
        <div class="summary-card">
            <div class="value">{total_columns}</div>
            <div class="label">Total Columns</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:{'#2ecc71' if avg_completeness > 90 else '#f39c12' if avg_completeness > 50 else '#e74c3c'}">{avg_completeness}%</div>
            <div class="label">Avg Completeness</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:{'#2ecc71' if avg_quality > 80 else '#f39c12' if avg_quality > 50 else '#e74c3c'}">{avg_quality if avg_quality else 'N/A'}</div>
            <div class="label">Avg Quality Score</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#e74c3c">{high_issues}</div>
            <div class="label">High Severity Issues</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#f39c12">{medium_issues}</div>
            <div class="label">Medium Issues</div>
        </div>
    </div>

    <h2>Visualizations</h2>
    {charts_html if charts_html else '<p>Charts not available (altair not installed)</p>'}

    {issues_table}

    <h2>Column Detail Cards</h2>
    {column_cards}

    <div class="footer">
        Silver Model Builder - AI-Powered Data Profiling Report<br>
        Generated with Snowflake Cortex AI
    </div>
</div>
</body>
</html>"""

    return html
