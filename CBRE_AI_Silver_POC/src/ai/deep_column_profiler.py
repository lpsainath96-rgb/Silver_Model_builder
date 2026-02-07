"""AI Deep Column Profiler.

Per-column LLM-driven analysis acting as a seasoned data profiling copilot.
Analyzes format patterns, quality issues, transformation recommendations,
and prescribes additional SQL checks.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("SNOWFLAKE_LLM_MODEL", "llama3-70b")


def build_deep_analysis_prompt(
    table: str,
    col: str,
    datatype: str,
    profile: Dict[str, Any],
    description: str,
    sample_values: List[str],
    business_context: str = "",
    target_mapping: str = "",
) -> str:
    """Build a comprehensive prompt for deep column analysis."""

    type_cat = profile.get("type_category", "other")
    total = profile.get("total", 0)
    nulls = profile.get("nulls", 0)
    distinct = profile.get("distinct", 0)
    completeness = round((total - nulls) / total * 100, 1) if total else 0
    uniqueness = round(distinct / total * 100, 1) if total else 0

    # Type-specific analysis instructions
    type_instructions = ""
    if type_cat == "string":
        ss = profile.get("string_stats", {})
        type_instructions = f"""
STRING-SPECIFIC ANALYSIS:
- Length range: {ss.get('min_length')}-{ss.get('max_length')} (avg {ss.get('avg_length')})
- Case: {ss.get('upper_case_pct', 0)}% UPPER, {ss.get('lower_case_pct', 0)}% lower
- Empty strings: {ss.get('empty_string_count', 0)}
- Detected patterns: {json.dumps(profile.get('detected_patterns', []))}
Analyze: case convention consistency, delimiter presence, format patterns (codes, identifiers, free text), encoding issues.
"""
    elif type_cat == "numeric":
        rng = profile.get("range", {})
        type_instructions = f"""
NUMERIC-SPECIFIC ANALYSIS:
- Range: {rng.get('min')} to {rng.get('max')}
- Distinct values: {distinct}
Analyze: Is this a measure (continuous) or a coded dimension (few values)? Look for suspicious sentinel values (-1, 0, 9999). Assess scale/precision needs.
"""
    elif type_cat == "date":
        rng = profile.get("range", {})
        type_instructions = f"""
DATE-SPECIFIC ANALYSIS:
- Range: {rng.get('min')} to {rng.get('max')}
Analyze: Date range reasonableness, presence of future dates, granularity (day/month/year), timezone considerations.
"""

    # Top values context
    top_vals = profile.get("top_values", [])
    top_str = json.dumps(top_vals[:5]) if top_vals else "[]"

    biz_ctx = f"\nBusiness Context (from user): {business_context}" if business_context else ""

    return f"""You are a seasoned data profiler with 20 years of experience in enterprise data warehousing and Silver/Gold layer modeling. Analyze this column deeply and provide structured findings.

TABLE: {table}
COLUMN: {col}
DATA TYPE: {datatype} (category: {type_cat})

PROFILE STATISTICS:
- Total rows: {total:,}
- Completeness: {completeness}% ({nulls:,} nulls/placeholders)
- Uniqueness: {uniqueness}% ({distinct:,} distinct values)
- Top values: {top_str}

PREVIOUS DESCRIPTION: {description}
{type_instructions}
SAMPLE VALUES (up to 20):
{json.dumps(sample_values[:20])}
{biz_ctx}

RESPOND WITH VALID JSON ONLY (no markdown, no explanation outside the JSON):
{{
    "data_quality_score": <0-100 integer>,
    "quality_issues": [
        {{"issue": "<short title>", "severity": "high|medium|low", "description": "<explanation>"}}
    ],
    "format_analysis": {{
        "dominant_pattern": "<describe the main data format>",
        "case_convention": "UPPER|lower|Mixed|N/A",
        "detected_formats": ["<list detected formats like email, phone, code, etc.>"]
    }},
    "statistical_insights": {{
        "distribution_type": "uniform|skewed|categorical|binary|unique_key|sparse",
        "cardinality_assessment": "unique_key|high|medium|low",
        "outlier_indicators": "<describe any outlier signals>"
    }},
    "silver_recommendations": {{
        "recommended_datatype": "<optimal Snowflake type for Silver>",
        "recommended_name": "<clean snake_case Silver column name>",
        "transformation_logic": "<SQL expression or description>",
        "is_pk_candidate": <true|false>
    }},
    "additional_profiling_prescriptions": [
        {{"check_name": "<name>", "sql_template": "<SELECT ... FROM {{TABLE}} WHERE ...>", "rationale": "<why run this>"}}
    ],
    "business_interpretation": "<2-3 sentence business meaning>"
}}"""


def deep_profile_column(
    conn,
    table: str,
    col: str,
    datatype: str,
    profile: Dict[str, Any],
    description: str = "",
    sample_values: List[str] = None,
    business_context: str = "",
) -> Dict[str, Any]:
    """Run AI deep analysis on a single column."""

    prompt = build_deep_analysis_prompt(
        table, col, datatype, profile, description,
        sample_values or profile.get("sample_values", []),
        business_context
    )

    try:
        escaped_prompt = prompt.replace("'", "''")
        sql = f"SELECT AI_COMPLETE('{LLM_MODEL}', '{escaped_prompt}') AS response"

        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            content = row[0] if row else None

        if not content:
            return _fallback_result(col, "Empty LLM response")

        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)

        result = json.loads(content)
        result["_column"] = col
        result["_status"] = "success"
        return result

    except json.JSONDecodeError as e:
        return _fallback_result(col, f"JSON parse error: {e}")
    except Exception as e:
        return _fallback_result(col, str(e))


def _fallback_result(col: str, error: str) -> Dict[str, Any]:
    """Return a default structure when AI analysis fails."""
    return {
        "_column": col,
        "_status": "error",
        "_error": error,
        "data_quality_score": None,
        "quality_issues": [],
        "format_analysis": {"dominant_pattern": "Unknown", "case_convention": "N/A", "detected_formats": []},
        "statistical_insights": {"distribution_type": "unknown", "cardinality_assessment": "unknown", "outlier_indicators": "N/A"},
        "silver_recommendations": {"recommended_datatype": None, "recommended_name": col.lower(), "transformation_logic": "Direct copy", "is_pk_candidate": False},
        "additional_profiling_prescriptions": [],
        "business_interpretation": f"Analysis failed: {error}",
    }


def deep_profile_table(
    conn,
    table: str,
    profiles: Dict[str, Dict[str, Any]],
    descriptions: Dict[str, str],
    sample_data: Dict[str, List[str]],
    business_contexts: Dict[str, str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run deep profiling on all columns in a table."""
    business_contexts = business_contexts or {}
    results = {}

    for col_name, prof in profiles.items():
        desc = descriptions.get(col_name, "")
        samples = sample_data.get(col_name.upper(), prof.get("sample_values", []))
        biz_ctx = business_contexts.get(col_name, "")

        result = deep_profile_column(
            conn, table, col_name, prof.get("datatype", "VARCHAR"),
            prof, desc, samples, biz_ctx
        )
        results[col_name] = result

    return results


def execute_prescribed_checks(
    conn,
    table: str,
    prescriptions: List[Dict[str, str]],
    schema: str = "PUBLIC",
) -> List[Dict[str, Any]]:
    """Safely execute AI-prescribed read-only SQL checks.

    Only SELECT/WITH statements are allowed. DDL/DML is rejected.
    """
    full_table = f'"{schema.upper()}"."{table}"'
    results = []

    for rx in prescriptions:
        check_name = rx.get("check_name", "unnamed")
        sql_template = rx.get("sql_template", "")
        rationale = rx.get("rationale", "")

        # Sanitize: only allow SELECT / WITH
        sql_clean = sql_template.strip().upper()
        if not (sql_clean.startswith("SELECT") or sql_clean.startswith("WITH")):
            results.append({
                "check_name": check_name,
                "status": "blocked",
                "reason": "Only SELECT/WITH queries are allowed",
                "sql": sql_template,
            })
            continue

        # Block dangerous keywords
        dangerous = re.findall(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|EXEC|CALL)\b",
            sql_clean
        )
        if dangerous:
            results.append({
                "check_name": check_name,
                "status": "blocked",
                "reason": f"Blocked keywords found: {dangerous}",
                "sql": sql_template,
            })
            continue

        # Substitute {TABLE} placeholder
        final_sql = sql_template.replace("{TABLE}", full_table)

        try:
            with conn.cursor() as cur:
                cur.execute(final_sql)
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
                data = [dict(zip(col_names, r)) for r in rows[:50]]

            results.append({
                "check_name": check_name,
                "status": "success",
                "rationale": rationale,
                "sql": final_sql,
                "row_count": len(rows),
                "data": data,
            })
        except Exception as e:
            results.append({
                "check_name": check_name,
                "status": "error",
                "error": str(e),
                "sql": final_sql,
            })

    return results
