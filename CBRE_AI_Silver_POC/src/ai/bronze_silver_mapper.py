"""Bronze-to-Silver Column Mapper using LLM Intelligence.

Uses Snowflake Cortex AI_COMPLETE to intelligently map Bronze source columns
to Silver target columns based on metadata, descriptions, and business context.
"""

import os
import json
import traceback
from typing import List, Dict, Any
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

# Snowflake connection parameters
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")

# LLM Model
LLM_MODEL = os.getenv("SNOWFLAKE_LLM_MODEL", "llama3.1-70b")


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
    )


def safe_get_description(descriptions: Any, table: str, col: str) -> str:
    """Safely extract a description from the descriptions structure."""
    if descriptions is None:
        return ""
    if not isinstance(descriptions, dict):
        return ""
    table_data = descriptions.get(table)
    if table_data is None:
        return ""
    if isinstance(table_data, str):
        # The table itself is a description string
        return table_data[:100]
    if isinstance(table_data, dict):
        col_desc = table_data.get(col, "")
        if isinstance(col_desc, str):
            return col_desc[:100]
        return ""
    return ""


def build_column_summary(table: str, col: str, profile: Any, description: str = "") -> str:
    """Build a summary of a column for the LLM prompt."""
    try:
        # Handle case where profile might be a string or other non-dict
        if not isinstance(profile, dict):
            desc_part = f" | Desc: {description}" if description else ""
            return f"  - {table}.{col} (Type: UNKNOWN){desc_part}"
        
        dtype = profile.get("datatype", "UNKNOWN") if isinstance(profile.get("datatype"), str) else "UNKNOWN"
        distinct = profile.get("distinct", 0)
        top_vals = profile.get("top_values", [])
        
        # Handle top_values which can be dicts or strings
        top_str_parts = []
        if top_vals and isinstance(top_vals, list):
            for v in top_vals[:3]:
                try:
                    if isinstance(v, dict):
                        val = v.get("value", "")
                        top_str_parts.append(str(val)[:30])
                    else:
                        top_str_parts.append(str(v)[:30])
                except Exception:
                    pass
        top_str = ", ".join(top_str_parts) if top_str_parts else "N/A"
        
        summary = f"  - {table}.{col} (Type: {dtype}, Distinct: {distinct}, Examples: {top_str})"
        if description:
            summary += f" | Desc: {description}"
        return summary
    except Exception as e:
        print(f"[bronze_silver_mapper] Error in build_column_summary for {table}.{col}: {e}")
        return f"  - {table}.{col} (Type: UNKNOWN)"


def map_bronze_to_silver(
    bronze_profiles: Dict[str, Dict[str, Any]],
    silver_profiles: Dict[str, Dict[str, Any]],
    bronze_descriptions: Any = None,
    silver_descriptions: Any = None,
    business_context: str = "",
    conn=None
) -> List[Dict[str, Any]]:
    """
    Map Bronze columns to Silver columns using LLM intelligence.
    """
    print(f"[bronze_silver_mapper] === STARTING MAPPING ===")
    print(f"[bronze_silver_mapper] bronze_profiles type: {type(bronze_profiles)}")
    print(f"[bronze_silver_mapper] silver_profiles type: {type(silver_profiles)}")
    print(f"[bronze_silver_mapper] bronze_descriptions type: {type(bronze_descriptions)}")
    
    # Build Bronze column list
    bronze_lines = ["BRONZE (Source) Columns:"]
    bronze_keys = []  # [(table, col, dtype)]
    
    try:
        if isinstance(bronze_profiles, dict):
            print(f"[bronze_silver_mapper] Bronze tables: {list(bronze_profiles.keys())}")
            for table, cols in bronze_profiles.items():
                print(f"[bronze_silver_mapper] Bronze table '{table}': cols type={type(cols).__name__}")
                if not isinstance(cols, dict):
                    print(f"[bronze_silver_mapper] SKIP: '{table}' cols is {type(cols).__name__}, not dict")
                    continue
                for col, profile in cols.items():
                    try:
                        desc = safe_get_description(bronze_descriptions, table, col)
                        bronze_lines.append(build_column_summary(table, col, profile, desc))
                        dtype = profile.get("datatype", "UNKNOWN") if isinstance(profile, dict) else "UNKNOWN"
                        bronze_keys.append((table, col, dtype))
                    except Exception as e:
                        print(f"[bronze_silver_mapper] Error processing Bronze {table}.{col}: {e}")
    except Exception as e:
        print(f"[bronze_silver_mapper] Error iterating bronze_profiles: {e}")
        traceback.print_exc()
    
    # Build Silver column list
    silver_lines = ["SILVER (Target) Columns:"]
    silver_keys = []
    
    try:
        if isinstance(silver_profiles, dict):
            print(f"[bronze_silver_mapper] Silver tables: {list(silver_profiles.keys())}")
            for table, cols in silver_profiles.items():
                print(f"[bronze_silver_mapper] Silver table '{table}': cols type={type(cols).__name__}")
                if not isinstance(cols, dict):
                    print(f"[bronze_silver_mapper] SKIP: '{table}' cols is {type(cols).__name__}, not dict")
                    continue
                for col, profile in cols.items():
                    try:
                        desc = safe_get_description(silver_descriptions, table, col)
                        silver_lines.append(build_column_summary(table, col, profile, desc))
                        dtype = profile.get("datatype", "UNKNOWN") if isinstance(profile, dict) else "UNKNOWN"
                        silver_keys.append((table, col, dtype))
                    except Exception as e:
                        print(f"[bronze_silver_mapper] Error processing Silver {table}.{col}: {e}")
    except Exception as e:
        print(f"[bronze_silver_mapper] Error iterating silver_profiles: {e}")
        traceback.print_exc()

    print(f"[bronze_silver_mapper] Bronze keys count: {len(bronze_keys)}")
    print(f"[bronze_silver_mapper] Silver keys count: {len(silver_keys)}")
    
    if not bronze_keys or not silver_keys:
        print(f"[bronze_silver_mapper] EARLY EXIT: No keys to map!")
        return []
    
    # Build LLM prompt
    prompt = f"""You are a data architect mapping Bronze (source) columns to Silver (target) columns.

{chr(10).join(bronze_lines)}

{chr(10).join(silver_lines)}

BUSINESS CONTEXT FROM USER:
{business_context if business_context else "No specific context provided."}

TASK:
For EACH Bronze column, determine the best matching Silver column based on:
1. Column names and naming patterns
2. Data types (compatibility)
3. Sample values and distributions
4. Business context provided

OUTPUT FORMAT (STRICT JSON):
Return a JSON array where each object has:
- "source_table": Bronze table name
- "source_column": Bronze column name
- "target_table": Silver table name (or "UNMAPPED" if no match)
- "target_column": Silver column name (or "UNMAPPED" if no match)
- "confidence": 0-100 integer (how confident in the mapping)
- "rationale": Brief explanation of why this mapping was chosen
- "transformation": Suggested transformation if needed (e.g., "CAST to VARCHAR", "Direct", "UPPER()", etc.)

Return ONLY the JSON array, no additional text.
"""

    print(f"[bronze_silver_mapper] Calling AI_COMPLETE with {len(bronze_keys)} bronze cols -> {len(silver_keys)} silver cols")
    
    # Call LLM
    close_conn = False
    if conn is None:
        conn = get_snowflake_connection()
        close_conn = True
    
    try:
        escaped_prompt = prompt.replace("'", "''")
        sql = f"SELECT AI_COMPLETE('{LLM_MODEL}', '{escaped_prompt}') AS response"
        
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            content = row[0] if row else "[]"
        
        print(f"[bronze_silver_mapper] AI_COMPLETE returned {len(content)} chars")
        print(f"[bronze_silver_mapper] Raw content preview: {content[:200]}")
        
        # Parse LLM response - handle double-stringified JSON
        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Parse JSON - may need multiple passes if LLM double-stringified
            mappings_raw = json.loads(content.strip())
            
            # If result is still a string, try parsing again (double-stringified case)
            parse_attempts = 0
            while isinstance(mappings_raw, str) and parse_attempts < 3:
                print(f"[bronze_silver_mapper] Result is string, parsing attempt {parse_attempts + 1}")
                mappings_raw = json.loads(mappings_raw)
                parse_attempts += 1
            
            print(f"[bronze_silver_mapper] Final mappings_raw type: {type(mappings_raw).__name__}")
            if isinstance(mappings_raw, list):
                print(f"[bronze_silver_mapper] Parsed {len(mappings_raw)} mappings from LLM response")
            else:
                print(f"[bronze_silver_mapper] WARNING: mappings_raw is not a list!")
                mappings_raw = []
                
        except json.JSONDecodeError as e:
            print(f"[bronze_silver_mapper] JSON parse error: {e}")
            print(f"[bronze_silver_mapper] Raw response: {content[:500]}")
            mappings_raw = []
        
        # Convert to standard format
        mappings = []

        if isinstance(mappings_raw, list):
            for i, m in enumerate(mappings_raw):
                print(f"[bronze_silver_mapper] Item {i}: type={type(m).__name__}")
                try:
                    if isinstance(m, str):
                        # Try to parse as JSON if it's a string
                        try:
                            m = json.loads(m)
                        except:
                            print(f"[bronze_silver_mapper] SKIP: Item {i} is unparseable string: {m[:100]}")
                            continue
                    
                    if not isinstance(m, dict):
                        print(f"[bronze_silver_mapper] SKIP: Item {i} is not dict: {type(m).__name__}")
                        continue
                    
                    mappings.append({
                        "SourceTable": m.get("source_table", "UNKNOWN"),
                        "SourceColumn": m.get("source_column", "UNKNOWN"),
                        "SourceDataType": next((k[2] for k in bronze_keys if k[0] == m.get("source_table") and k[1] == m.get("source_column")), "UNKNOWN"),
                        "TargetTable": m.get("target_table", "UNMAPPED"),
                        "TargetColumn": m.get("target_column", "UNMAPPED"),
                        "TargetDataType": next((k[2] for k in silver_keys if k[0] == m.get("target_table") and k[1] == m.get("target_column")), ""),
                        "Confidence": m.get("confidence", 0),
                        "MappingRationale": m.get("rationale", ""),
                        "SuggestedTransformation": m.get("transformation", "Direct")
                    })
                except Exception as e:
                    print(f"[bronze_silver_mapper] Error converting mapping {i}: {e}")
        else:
            print(f"[bronze_silver_mapper] ERROR: mappings_raw is not a list, it's: {type(mappings_raw).__name__}")

        
        # Sort by confidence descending
        mappings.sort(key=lambda x: x["Confidence"], reverse=True)
        
        print(f"[bronze_silver_mapper] Generated {len(mappings)} final mappings")
        return mappings
        
    except Exception as e:
        print(f"[bronze_silver_mapper] Error: {e}")
        traceback.print_exc()
        return []
    finally:
        if close_conn and conn:
            conn.close()


if __name__ == "__main__":
    print("Bronze-Silver LLM Mapper module loaded successfully.")
