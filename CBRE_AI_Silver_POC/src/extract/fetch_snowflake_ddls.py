import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import snowflake.connector

load_dotenv(Path(__file__).resolve().parents[2] / "src" / ".env")

def get_connection():
    raw_account = os.getenv("SNOWFLAKE_ACCOUNT") or ""
    # Normalize account (Snowflake connector expects account locator, not full domain)
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

def fetch_table_ddl(table_name: str) -> str:
    """Returns full CREATE TABLE statement including clustering & comments."""
    sql = f"SHOW CREATE TABLE {table_name}"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            # SHOW CREATE TABLE returns: name, kind, ddl, etc. DDL commonly at index 2
            return row[2] if row and len(row) >= 3 else ""

def list_tables(limit: int = 4) -> List[str]:
    """Return up to `limit` table names from the current schema (alphabetical)."""
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = %s ORDER BY table_name"
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    names: List[str] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (schema,))
            for i, (name,) in enumerate(cur.fetchall()):
                if i >= limit:
                    break
                names.append(name)
    return names

def fetch_multiple_ddls(tables: List[str]) -> Dict[str, str]:
    return {t: fetch_table_ddl(t) for t in tables}

def save_ddls(ddls: Dict[str, str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for tbl, ddl in ddls.items():
            f.write(f"-- DDL for {tbl}\n{ddl};\n\n")

if __name__ == "__main__":
    tables = list_tables(limit=4)
    if not tables:
        raise SystemExit("No tables found in schema; verify SNOWFLAKE_SCHEMA and permissions.")
    print(f"[INFO] Fetching DDLs for tables: {', '.join(tables)}")
    ddls = fetch_multiple_ddls(tables)
    out_file = Path("data/V4-Demo/snowflake_extracted_ddls.sql")
    save_ddls(ddls, out_file)
    for t, ddl in ddls.items():
        print(f"[OK] {t} DDL length: {len(ddl)}")
    print(f"[DONE] Saved DDLs to {out_file}")