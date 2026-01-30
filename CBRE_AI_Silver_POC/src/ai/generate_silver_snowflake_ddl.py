"""Generate Snowflake DDL from a Silver model JSON.

Enhancements:
 - Robust JSON parsing (handles Markdown and string concatenation).
 - DIM_ / FACT_ naming support.
 - VARIANT and ARRAY type support for performance.
 - Database and Schema defaults from environment.
"""
import json, re, argparse, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GENERIC_TYPE_MAP = {
    "string": "VARCHAR",
    "numeric": "NUMBER",
    "date": "DATE",
    "boolean": "BOOLEAN",
    "float": "FLOAT",
    "int": "NUMBER",
    "timestamp": "TIMESTAMP_NTZ",
    "json": "VARIANT",
    "object": "VARIANT",
    "array": "ARRAY",
}

DEFAULT_PK_CANDIDATES = {"COMPANY_ID", "OPPORTUNITY_ID", "YEAR", "CLIENT_ID"}

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).upper()

def normalize_type(dt: str):
    up = dt.upper()
    if "VARIANT" in up: return "VARIANT"
    if "ARRAY" in up: return "ARRAY"
    if "OBJECT" in up: return "VARIANT"
    if "JSON" in up: return "VARIANT"
    
    # Pass through explicit NUMBER/NUMERIC(X,Y)
    if re.match(r'(NUMBER|NUMERIC)\(\d+,\d+\)', up):
        return up
    if re.match(r'(NUMBER|NUMERIC)\(\d+\)', up):
        return up
    if "DATE" in up: return "DATE"
    if "TIMESTAMP" in up: return "TIMESTAMP_NTZ"
    if any(k in up for k in ("CHAR","TEXT","STRING","VARCHAR")): return "VARCHAR"
    if any(k in up for k in ("INT","NUMBER","NUMERIC","DECIMAL")): return "NUMBER"
    return "VARCHAR"

def build_column_line(col_name: str, attr: dict) -> str:
    dtype = normalize_type(attr.get("datatype","VARCHAR"))
    return f'    "{col_name}" {dtype}'

def parse_robust(text):
    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Pre-clean: common LLM hallucinated JSON artifacts
    # 1. Backslash at end of line (continuation)
    text = re.sub(r'\\\n', ' ', text)
    
    # Try extracting from Markdown code blocks
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        blob = match.group(1).strip()
        # Fix Javascript-style string concatenation: "..." + "..."
        blob = re.sub(r'"\s*\+\s*"', '', blob)
        # Fix backslash continuations inside the blob too
        blob = re.sub(r'\\\n', ' ', blob)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
            
    # Try finding the first { and last }
    p1 = text.find('{')
    p2 = text.rfind('}')
    if p1 >= 0 and p2 > p1:
        blob = text[p1:p2+1]
        blob = re.sub(r'"\s*\+\s*"', '', blob)
        blob = re.sub(r'\\\n', ' ', blob)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate Snowflake DDL from Silver model JSON")
    base = Path(__file__).resolve().parent.parent  # src/
    default_in = base / "data" / "real_run" / "silver_model_draft.json"
    default_out = base / "data" / "real_run" / "silver_model_DDL.sql"
    
    default_db = os.getenv("SNOWFLAKE_DATABASE", "PROD_C360_DB")
    default_schema = os.getenv("SNOWFLAKE_SCHEMA", "SILVER")
    
    parser.add_argument("-i", "--input", default=str(default_in))
    parser.add_argument("-o", "--output", default=str(default_out))
    parser.add_argument("--db", default=default_db)
    parser.add_argument("--schema", default=default_schema)
    parser.add_argument("--include-comments", action="store_true", help="Emit COMMENT statements for descriptions")
    parser.add_argument("--transient", action="store_true", help="Use TRANSIENT TABLE instead of regular TABLE")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        exit(1)
        
    raw_content = input_file.read_text(encoding='utf-8')
    data = parse_robust(raw_content)
    
    # Handle nested strings (GPT sometimes double-wraps)
    if isinstance(data, str):
        data = parse_robust(data)
            
    if not isinstance(data, dict):
        print(f"ERROR: Could not parse valid JSON object from {input_file}")
        exit(1)

    ddl_statements = []
    comment_statements = []

    for entity in data.get("entities", []):
        table_name = sanitize(entity.get("entity_name", "UNKNOWN"))
        attrs = []
        seen = set()
        pk_cols = []
        
        for attr in entity.get("attributes", []):
            name = attr.get("name")
            dtype = attr.get("datatype")
            if not name or not dtype:
                continue
                
            col = sanitize(name)
            # Handle duplicate column names (rare but possible in AI output)
            if col in seen:
                i = 2
                while f"{col}_{i}" in seen:
                    i += 1
                col = f"{col}_{i}"
            seen.add(col)
            
            # PK detection
            if attr.get("is_pk") or col in DEFAULT_PK_CANDIDATES:
                pk_cols.append(col)
                
            attrs.append(build_column_line(col, attr))
            
            # Comments
            if args.include_comments and attr.get("description"):
                desc_safe = attr['description'].replace("'", "''")
                comment_statements.append(
                    f"COMMENT ON COLUMN {args.db}.{args.schema}.{table_name}.\"{col}\" IS '{desc_safe}';"
                )
                
        if not attrs:
            continue
            
        # PK Clause
        if pk_cols:
             quoted_pks = [f'"{c}"' for c in pk_cols]
             pk_clause = f",\n    PRIMARY KEY ({', '.join(quoted_pks)})"
        else:
             pk_clause = ""
             
        table_keyword = "TRANSIENT TABLE" if args.transient else "TABLE"
        full_table_name = f"{args.db}.{args.schema}.{table_name}"
        
        ddl = f'CREATE OR REPLACE {table_keyword} {full_table_name} (\n' + ",\n".join(attrs) + pk_clause + "\n);"
        ddl_statements.append(ddl)
        
    if args.include_comments and comment_statements:
        ddl_statements.extend(comment_statements)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(ddl_statements), encoding='utf-8')
    print(f"Wrote Snowflake DDL -> {out_path}")

if __name__ == "__main__":
    main()