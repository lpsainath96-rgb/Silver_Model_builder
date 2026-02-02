"""Real data profiler for Snowflake tables.

Profiles columns directly from Snowflake tables using aggregation queries,
outputting JSON compatible with the existing AI pipeline.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import snowflake.connector

load_dotenv() # Load from standard .env locations


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
        cur.execute(sql, (schema, table_name.upper()))
        return [{"name": row[0], "datatype": row[1]} for row in cur.fetchall()]


def profile_column(conn, table_name: str, column_name: str, datatype: str) -> Dict[str, Any]:
    """Profile a single column using Snowflake aggregation queries."""
    schema = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
    full_table = f'"{schema}"."{table_name.upper()}"'
    col = f'"{column_name.upper()}"'
    
    # Basic stats query (using APPROX_COUNT_DISTINCT for speed on large tables)
    stats_sql = f"""
        SELECT 
            COUNT(*) AS total,
            COUNT({col}) AS non_null,
            COUNT(*) - COUNT({col}) AS nulls,
            APPROX_COUNT_DISTINCT({col}) AS distinct_count
        FROM {full_table}
    """
    
    with conn.cursor() as cur:
        cur.execute(stats_sql)
        row = cur.fetchone()
        total, non_null, nulls, distinct = row[0], row[1], row[2], row[3]
    
    profile = {
        "datatype": datatype,
        "total": total,
        "nulls": nulls,
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
            profile["top_values"] = [str(row[0]) for row in cur.fetchall()]
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


def profile_table(conn, table_name: str, target_columns: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Profile all (or specific) columns in a table."""
    columns = get_column_info(conn, table_name)
    table_profile = {}
    
    # Filter if target_columns is provided
    if target_columns:
        # Normalize to uppercase for case-insensitive matching
        target_set = {c.upper() for c in target_columns}
        columns = [c for c in columns if c["name"].upper() in target_set]

    for col in columns:
        col_name = col["name"]
        datatype = col["datatype"]
        print(f"    Profiling column: {col_name}")
        table_profile[col_name] = profile_column(conn, table_name, col_name, datatype)
    
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
