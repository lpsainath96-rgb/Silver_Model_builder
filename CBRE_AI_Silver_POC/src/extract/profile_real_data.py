"""Real data profiler for Snowflake tables.

Profiles columns directly from Snowflake tables using aggregation queries,
outputting JSON compatible with the existing AI pipeline.
Includes extended string metrics, pattern detection, and expanded sample values.
"""

import os
import json
import argparse
from pathlib import Path
from decimal import Decimal
from typing import Dict, List, Any
from dotenv import load_dotenv
import snowflake.connector


STRING_TYPES = ("VARCHAR", "TEXT", "STRING", "CHAR", "NVARCHAR", "NCHAR")


def to_json_safe(val):
    """Convert Decimal or other non-JSON types to standard Python types."""
    if isinstance(val, Decimal):
        # Convert to float if it has decimals, otherwise int
        return float(val) if val % 1 else int(val)
    return val


def classify_column_type(datatype: str) -> str:
    """Classify a Snowflake data type into a broad category."""
    dt = datatype.upper()
    if dt.startswith(STRING_TYPES):
        return "string"
    if dt.startswith(("NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE", "REAL", "NUMERIC", "BIGINT", "SMALLINT", "TINYINT")):
        return "numeric"
    if dt.startswith(("DATE", "TIMESTAMP", "TIME")):
        return "date"
    if dt.startswith(("BOOLEAN",)):
        return "boolean"
    if dt.startswith(("VARIANT", "OBJECT", "ARRAY")):
        return "semi_structured"
    return "other"


def fetch_sample_rows(conn, full_table: str, columns: List[Dict[str, str]], n: int = 20, sample_clause: str = "") -> Dict[str, List[str]]:
    """Fetch up to n sample values per column using a single query."""
    result = {}
    col_refs = ", ".join([f'"{c["name"].upper()}"' for c in columns])
    try:
        sql = f'SELECT {col_refs} FROM {full_table}{sample_clause} WHERE TRUE LIMIT {n}'
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            col_names_out = [desc[0] for desc in cur.description]
            for i, cname in enumerate(col_names_out):
                vals = []
                for row in rows:
                    if row[i] is not None:
                        vals.append(str(row[i]))
                result[cname] = vals
    except Exception:
        for c in columns:
            result[c["name"].upper()] = []
    return result


def detect_string_patterns(conn, full_table: str, col: str, sample_clause: str = "") -> List[Dict[str, Any]]:
    """Detect common data patterns in a string column using REGEXP_LIKE."""
    c_ref = f'"{col}"'
    patterns = [
        ("email", r"'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'"),
        ("phone", r"'^[\\+]?[0-9\\s\\-\\(\\)]{7,15}$'"),
        ("url", r"'^https?://[^\\s]+$'"),
        ("numeric_string", r"'^[0-9]+\\.?[0-9]*$'"),
        ("code_pattern", r"'^[A-Z]{2,5}[-_][A-Z0-9]+$'"),
    ]
    results = []
    for pname, regex in patterns:
        try:
            sql = f"""
                SELECT
                    COUNT(CASE WHEN REGEXP_LIKE(CAST({c_ref} AS STRING), {regex}) THEN 1 END) AS match_cnt,
                    COUNT({c_ref}) AS total_cnt
                FROM {full_table}{sample_clause}
                WHERE {c_ref} IS NOT NULL
            """
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                if row and row[1] and row[1] > 0:
                    pct = round(row[0] / row[1] * 100, 1)
                    if pct > 1.0:
                        results.append({"pattern": pname, "match_pct": pct})
        except Exception:
            continue
    return results


def get_connection():
    """Create a Snowflake connection using environment variables."""
    raw_account = os.getenv("SNOWFLAKE_ACCOUNT") or ""
    if raw_account.endswith(".snowflakecomputing.com"):
        raw_account = raw_account.replace(".snowflakecomputing.com", "")
    return snowflake.connector.connect(
        account=raw_account,
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


def list_tables(conn, limit: int = 10) -> List[str]:
    """List tables in the current schema."""
    schema = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
    sql = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = %s 
        ORDER BY table_name
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema,))
        tables = [row[0] for i, row in enumerate(cur.fetchall()) if i < limit]
    return tables


def get_column_info(conn, table_name: str) -> List[Dict[str, str]]:
    """Get column names and data types for a table."""
    schema = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
    sql = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        # Respect the exact case of the table name as it was listed
        cur.execute(sql, (schema, table_name))
        cols = [{"name": row[0], "datatype": row[1]} for row in cur.fetchall()]
        if not cols:
            # Fallback to upper if exact match fails (Snowflake standard)
            cur.execute(sql, (schema, table_name.upper()))
            cols = [{"name": row[0], "datatype": row[1]} for row in cur.fetchall()]
        return cols


def profile_column(conn, table_name: str, column_name: str, datatype: str) -> Dict[str, Any]:
    """Profile a single column using Snowflake aggregation queries."""
    schema = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
    full_table = f'"{schema}"."{table_name.upper()}"'
    col = f'"{column_name.upper()}"'
    
    # Improved stats query: Treats physical NULLs, empty strings, whitespace, 
    # and common placeholders (N/A, NULL, NONE) as "Incomplete".
    stats_sql = f"""
        SELECT 
            COUNT(*) AS total,
            COUNT({col}) AS physical_valid,
            COUNT(CASE 
                WHEN {col} IS NULL THEN NULL
                WHEN TRIM(CAST({col} AS STRING)) = '' THEN NULL
                WHEN UPPER(TRIM(CAST({col} AS STRING))) IN ('N/A', 'NA', 'NULL', 'NONE', '<NULL>', '.') THEN NULL
                ELSE 1 
            END) AS robust_valid,
            APPROX_COUNT_DISTINCT({col}) AS distinct_count
        FROM {full_table}
    """
    
    with conn.cursor() as cur:
        cur.execute(stats_sql)
        row = cur.fetchone()
        total, phys_valid, rob_valid, distinct = row[0], row[1], row[2], row[3]
    
    profile = {
        "datatype": datatype,
        "total": total,
        "nulls": total - rob_valid, # Robust nulls (includes placeholders)
        "physical_nulls": total - phys_valid, # Standard database nulls
        "distinct": distinct,
        "top_values": [],
        "sample_values": [],
    }
    
    # Get top values (most frequent)
    try:
        top_sql = f"""
            SELECT {col}, COUNT(*) AS cnt 
            FROM {full_table} 
            WHERE {col} IS NOT NULL
            GROUP BY {col} 
            ORDER BY cnt DESC 
            LIMIT 3
        """
        with conn.cursor() as cur:
            cur.execute(top_sql)
            # Store as list of dicts for easier charting
            profile["top_values"] = [{"value": str(row[0]), "count": int(row[1])} for row in cur.fetchall()]
    except Exception:
        pass  # Some column types may not support GROUP BY
    
    # Get sample values
    try:
        sample_sql = f"""
            SELECT DISTINCT {col} 
            FROM {full_table} 
            WHERE {col} IS NOT NULL 
            LIMIT 2
        """
        with conn.cursor() as cur:
            cur.execute(sample_sql)
            profile["sample_values"] = [str(row[0]) for row in cur.fetchall()]
    except Exception:
        pass
    
    # Get min/max for numeric types
    numeric_types = ("NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE", "REAL")
    if datatype.upper().startswith(numeric_types):
        try:
            range_sql = f"""
                SELECT MIN({col}), MAX({col}) 
                FROM {full_table}
            """
            with conn.cursor() as cur:
                cur.execute(range_sql)
                row = cur.fetchone()
                if row[0] is not None:
                    profile["range"] = {
                        "min": float(row[0]),
                        "max": float(row[1])
                    }
        except Exception:
            pass
    
    return profile


def profile_table(conn, table_name: str, target_columns: List[str] = None, sample_pct: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    Profile a table efficiently by batching basic stats into a single wide query.
    Handles 'Millions of Records' by pushing all work to Snowflake's columnar engine.
    """
    schema = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
    # Use exact casing for table name
    full_table = f'"{schema}"."{table_name}"'
    
    # Get column definitions
    columns_info = get_column_info(conn, table_name)
    if target_columns:
        target_set = {c.upper() for c in target_columns}
        columns_info = [c for c in columns_info if c["name"].upper() in target_set]

    if not columns_info:
        print(f"Warning: No valid columns found for profiling table {table_name}")
        return {}

    # --- STEP 1: BATCH BASIC STATS ---
    select_clauses = ["COUNT(*) AS TOTAL_ROWS"]
    for col in columns_info:
        c_name = col["name"].upper()
        c_ref = f'"{c_name}"'

        # Robust valid logic
        select_clauses.append(f"""
            COUNT(CASE
                WHEN {c_ref} IS NULL THEN NULL
                WHEN TRIM(CAST({c_ref} AS STRING)) = '' THEN NULL
                WHEN UPPER(TRIM(CAST({c_ref} AS STRING))) IN ('N/A', 'NA', 'NULL', 'NONE', '<NULL>', '.') THEN NULL
                ELSE 1
            END) AS {c_name}_ROB_VALID""")

        select_clauses.append(f'COUNT({c_ref}) AS {c_name}_PHYS_VALID')
        select_clauses.append(f'APPROX_COUNT_DISTINCT({c_ref}) AS {c_name}_DISTINCT')

        numeric_types = ("NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE", "REAL", "DATE", "TIMESTAMP")
        if col["datatype"].upper().startswith(numeric_types):
             select_clauses.append(f'MIN({c_ref}) AS {c_name}_MIN')
             select_clauses.append(f'MAX({c_ref}) AS {c_name}_MAX')

        # String-type extended metrics
        if classify_column_type(col["datatype"]) == "string":
            select_clauses.append(f'MIN(LENGTH(TRIM(CAST({c_ref} AS STRING)))) AS {c_name}_MIN_LEN')
            select_clauses.append(f'MAX(LENGTH(TRIM(CAST({c_ref} AS STRING)))) AS {c_name}_MAX_LEN')
            select_clauses.append(f'AVG(LENGTH(TRIM(CAST({c_ref} AS STRING)))) AS {c_name}_AVG_LEN')
            select_clauses.append(f'SUM(CASE WHEN {c_ref} = UPPER({c_ref}) AND {c_ref} IS NOT NULL THEN 1 ELSE 0 END) AS {c_name}_UPPER_CNT')
            select_clauses.append(f'SUM(CASE WHEN {c_ref} = LOWER({c_ref}) AND {c_ref} IS NOT NULL THEN 1 ELSE 0 END) AS {c_name}_LOWER_CNT')
            select_clauses.append(f"SUM(CASE WHEN TRIM(CAST({c_ref} AS STRING)) = '' THEN 1 ELSE 0 END) AS {c_name}_EMPTY_CNT")

    sample_clause = f" TABLESAMPLE ({sample_pct})" if sample_pct < 100 else ""
    batch_sql = f"SELECT \n  " + ",\n  ".join(select_clauses) + f"\nFROM {full_table}{sample_clause}"
    
    table_profile = {}
    
    try:
        with conn.cursor() as cur:
            cur.execute(batch_sql)
            row = cur.fetchone()
            col_names = [desc[0] for desc in cur.description]
            # Convert all Snowflake types (like Decimal) to JSON-safe Python types
            stats_map = {name: to_json_safe(val) for name, val in zip(col_names, row)}
            total_rows = stats_map.get("TOTAL_ROWS", 0)
    except Exception as e:
        raise Exception(f"Batch profiling failed for {table_name}: {str(e)}")

    # --- STEP 1b: FETCH EXPANDED SAMPLE VALUES ---
    sample_data = fetch_sample_rows(conn, full_table, columns_info, n=20, sample_clause=sample_clause)

    # --- STEP 2: FINALIZE PROFILES ---
    for col in columns_info:
        c_name = col["name"].upper()
        dtype = col["datatype"]
        type_cat = classify_column_type(dtype)

        rob_valid = stats_map.get(f"{c_name}_ROB_VALID", 0)
        phys_valid = stats_map.get(f"{c_name}_PHYS_VALID", 0)
        distinct = stats_map.get(f"{c_name}_DISTINCT", 0)

        profile = {
            "datatype": dtype,
            "type_category": type_cat,
            "total": total_rows,
            "nulls": total_rows - rob_valid,
            "physical_nulls": total_rows - phys_valid,
            "distinct": distinct,
            "top_values": [],
            "sample_values": sample_data.get(c_name, []),
            "range": {}
        }

        if f"{c_name}_MIN" in stats_map:
            profile["range"] = {
                "min": stats_map[f"{c_name}_MIN"],
                "max": stats_map[f"{c_name}_MAX"]
            }

        # String-type extended stats
        if type_cat == "string":
            phys_nn = phys_valid or 1  # avoid division by zero
            upper_cnt = stats_map.get(f"{c_name}_UPPER_CNT", 0) or 0
            lower_cnt = stats_map.get(f"{c_name}_LOWER_CNT", 0) or 0
            profile["string_stats"] = {
                "min_length": to_json_safe(stats_map.get(f"{c_name}_MIN_LEN")),
                "max_length": to_json_safe(stats_map.get(f"{c_name}_MAX_LEN")),
                "avg_length": round(float(stats_map.get(f"{c_name}_AVG_LEN") or 0), 1),
                "upper_case_pct": round(upper_cnt / phys_nn * 100, 1),
                "lower_case_pct": round(lower_cnt / phys_nn * 100, 1),
                "empty_string_count": to_json_safe(stats_map.get(f"{c_name}_EMPTY_CNT", 0))
            }
            # Detect patterns for string columns
            profile["detected_patterns"] = detect_string_patterns(conn, full_table, c_name, sample_clause)

        # --- STEP 3: TOP VALUES (Required separate query per column) ---
        try:
            top_sql = f"""
                SELECT "{c_name}", COUNT(*) AS cnt
                FROM {full_table}{sample_clause}
                WHERE "{c_name}" IS NOT NULL
                GROUP BY 1 ORDER BY 2 DESC LIMIT 5
            """
            with conn.cursor() as cur:
                cur.execute(top_sql)
                profile["top_values"] = [{"value": str(r[0]), "count": to_json_safe(r[1])} for r in cur.fetchall()]
        except Exception:
            pass

        table_profile[col["name"]] = profile

    return table_profile


def main():
    parser = argparse.ArgumentParser(description="Profile real Snowflake data")
    parser.add_argument("-t", "--tables", nargs="*", help="Specific table names to profile")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Max tables to profile if not specified")
    parser.add_argument("-o", "--output", default="data/real_run/profile/column_profiles_real.json", help="Output JSON path")
    args = parser.parse_args()
    
    print("[profile_real_data] Connecting to Snowflake...")
    conn = get_connection()
    
    try:
        # Use uppercase schema for Information Schema queries
        schema = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
        
        if args.tables:
            tables = args.tables
        else:
            tables = list_tables(conn, limit=args.limit)
        
        print(f"[profile_real_data] Profiling {len(tables)} tables in schema {schema}: {', '.join(tables)}")
        
        profiles = {}
        for table in tables:
            try:
                print(f"  Profiling table: {table}")
                profiles[table] = profile_table(conn, table)
            except Exception as e:
                print(f"  ❌ Error profiling table {table}: {e}")
                continue
    finally:
        conn.close()
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, indent=2), encoding='utf-8')
    print(f"[profile_real_data] Profiles written to: {out_path}")

    # Generate and save compatible metadata.json
    metadata_out = {}
    for table, col_map in profiles.items():
        cols_list = []
        for col_name, stats in col_map.items():
            cols_list.append({
                "name": col_name,
                "datatype": stats.get("datatype", "UNKNOWN")
            })
        metadata_out[table] = {"columns": cols_list}
    
    meta_path = out_path.parent / "real_data_metadata.json"
    meta_path.write_text(json.dumps(metadata_out, indent=2), encoding='utf-8')
    print(f"[profile_real_data] Metadata written to: {meta_path}")


if __name__ == "__main__":
    main()
