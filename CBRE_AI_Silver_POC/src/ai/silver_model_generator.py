try:
    from openai import AzureOpenAI, OpenAI  # retained for potential future use
except ImportError:
    AzureOpenAI = None
import json, os
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Snowflake connection parameters
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# LLM model to use with AI_COMPLETE
LLM_MODEL = os.getenv("SNOWFLAKE_LLM_MODEL", "llama3-70b")


def get_snowflake_connection():
    """Create a Snowflake connection."""
    account = SNOWFLAKE_ACCOUNT
    if account.endswith(".snowflakecomputing.com"):
        account = account.replace(".snowflakecomputing.com", "")
    return snowflake.connector.connect(
        account=account,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )


def generate_silver_model(clustered_data):
    print("[silver_model_generator] Enter generate_silver_model; preparing prompt.")
    
    # Use real DB and Schema for the prompt
    target_db = SNOWFLAKE_DATABASE or "PROD_C360_DB"
    target_schema = "SILVER"  # Explicitly move to SILVER schema
    
    system_prompt = f"""
You are a senior data architect designing Silver-layer data models in Snowflake.

CONTEXT
- Input: Column clusters grouped by semantic similarity from Bronze sources.
- Database: {target_db}
- Schema: {target_schema}

CRITICAL COLUMN MAPPING RULES:
1. ONE-TO-ONE MAPPING: Create ONE explicit Silver target column for EACH bronze source column.
   - DO NOT drop any columns.
   - DO NOT group multiple columns into one (unless semantically identical, e.g., CUSTOMER_ID and CLIENT_ID can merge).
   
2. VARIANT DATATYPE:
   - ONLY use VARIANT for source columns that are already semi-structured (VARIANT, OBJECT, ARRAY, JSON).
   - DO NOT use VARIANT to consolidate regular VARCHAR/NUMBER/DATE columns.

3. ENTITY DESIGN:
   - Use "DIM_*" for Master/Reference data.
   - Use "FACT_*" for transactional data.

4. DATATYPE MAPPING:
   - VARCHAR -> VARCHAR(16777216)
   - NUMBER/FLOAT/DECIMAL -> NUMBER(38,X)
   - DATE/TIMESTAMP -> DATE or TIMESTAMP_NTZ
   - VARIANT/OBJECT/ARRAY -> VARIANT

DDL STYLE:
- Fully qualify table names: {target_db}.{target_schema}.<ENTITY_NAME>
- Use TRANSIENT TABLE.

OUTPUT FORMAT (STRICT JSON):
{{
  "entities": [
    {{
      "entity_name": "DIM_COMPANY",
      "purpose": "Company master data",
      "attributes": [
        {{
          "name": "COMPANY_ID",
          "datatype": "VARCHAR(16777216)",
          "nullable": false,
          "is_pk": true,
          "source_columns": ["BRONZE.TABLE.COMPANY_ID"],
          "rationale": "Primary key, 1:1 mapping."
        }},
        {{
          "name": "COMPANY_NAME",
          "datatype": "VARCHAR(16777216)",
          "nullable": true,
          "source_columns": ["BRONZE.TABLE.NAME"],
          "rationale": "Business name, 1:1 mapping."
        }}
      ]
    }}
  ]
}}
"""

    prompt = f"Column clusters JSON:\n{json.dumps(clustered_data, indent=2)[:30000]}"  # Increased context window
    print("[silver_model_generator] Prompt length:", len(prompt))
    print(f"[silver_model_generator] Using Snowflake Cortex model: {LLM_MODEL}")
 
    try:
        conn = get_snowflake_connection()
        # Combine system and user prompt for Cortex (Llama models often prefer single prompt blocks)
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n{prompt}"
        
        # Escape single quotes and call Snowflake
        escaped_prompt = full_prompt.replace("'", "''")
        sql = f"SELECT AI_COMPLETE('{LLM_MODEL}', '{escaped_prompt}') AS response"
        
        print(f"[silver_model_generator] Executing Snowflake AI_COMPLETE...")
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            content = row[0] if row else None
            
        conn.close()
        
        if not content:
            return json.dumps({"entities": [], "error": "Empty response from Snowflake Cortex"}, indent=2)
            
    except Exception as e:
        print(f"[silver_model_generator] Snowflake Cortex error: {e}")
        return json.dumps({"entities": [], "error": str(e)}, indent=2)

    print("[silver_model_generator] Received content length:", len(content))
    return content
 
 
if __name__ == "__main__":
    import argparse
    base = Path(__file__).resolve().parents[2]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=str(base / "data" / "real_run" / "profile" / "semantic_clusters.json"))
    parser.add_argument("-o", "--output", default=str(base / "data" / "real_run" / "silver_model_draft.json"))
    args = parser.parse_args()

    clusters_path = Path(args.input)
    print(f"[silver_model_generator] Using clusters file: {clusters_path}")
    if not clusters_path.exists():
        raise SystemExit(f"Cluster file not found: {clusters_path}")
    
    clustered_data = json.loads(clusters_path.read_text(encoding='utf-8'))
    print(f"[silver_model_generator] Loaded clusters with keys: {list(clustered_data.keys())}")
    
    result = generate_silver_model(clustered_data)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[silver_model_generator] Writing output to: {output_path}")
    output_path.write_text(result, encoding='utf-8')
    print(f"Silver model draft saved to {output_path}")