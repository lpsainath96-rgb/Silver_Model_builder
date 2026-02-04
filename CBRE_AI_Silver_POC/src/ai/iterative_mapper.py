"""Iterative Bronze-to-Silver Column Mapper with Structured JSON Output.

Uses Snowflake Cortex AI_COMPLETE with JSON schema for reliable structured output.
Implements iterative batch processing for scalable large-schema mapping.
"""

import os
import json
import traceback
from typing import List, Dict, Any, Optional
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

# LLM Model - using openai-gpt-5.1 for mapping
LLM_MODEL = os.getenv("SNOWFLAKE_LLM_MODEL", "openai-gpt-5.1")



# Batch sizes
TARGET_BATCH_SIZE = 5   # Number of target columns per batch
SOURCE_CHUNK_SIZE = 10  # Reduced from 15 to prevent truncation


# --- JSON SCHEMA FOR STRUCTURED OUTPUT ---

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "mapped_columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_column_name": {"type": "string"},
                    "source_column_name": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_column_name": {"type": "string"},
                                "source_table_name": {"type": "string"},
                                "source_schema_name": {"type": "string"},
                                "justification": {"type": "string"}
                            }
                        }
                    },
                    "final_transformation_logic": {"type": "string"},
                    "work_notes": {"type": "string"}
                },
                "required": ["target_column_name", "source_column_name", "final_transformation_logic", "work_notes"]
            }
        }
    },
    "required": ["mapped_columns"]
}


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


# --- DESCRIPTION GENERATION ---

# --- SEMANTIC HINTS (from generate_long_descriptions.py) ---
SEMANTIC_HINTS = [
    (lambda n: n.endswith('_DT') or 'DATE' in n or 'TIMESTAMP' in n, 'timestamp of record ingestion or event occurrence'),
    (lambda n: 'YEAR' in n, 'calendar year for the financial measure or dimension context'),
    (lambda n: 'MONTH' in n, 'calendar month name (English) representing period granularity'),
    (lambda n: 'COUNTRY' in n, 'ISO country code identifying reporting geography'),
    (lambda n: 'CURRENCY' in n, 'three-letter currency code (ISO 4217)'),
    (lambda n: 'VALUE' in n and 'HASH' not in n, 'monetary amount; numeric financial measure'),
    (lambda n: 'MD5_HASH' in n or 'HASH' in n, 'MD5 hash fingerprint used for change detection and deduplication'),
    (lambda n: 'FUNCTION' in n, 'organizational functional unit classification'),
    (lambda n: 'MANAGING_OFFICE' in n, 'managing office location identifier'),
    (lambda n: 'LOB' in n or 'DIVISION' in n, 'line of business / hierarchical reporting segment'),
    (lambda n: 'CLIENT' in n, 'client identifier at given aggregation level'),
]

def infer_semantic(col_name: str) -> str:
    """Infer semantic meaning from column name patterns."""
    upper = col_name.upper()
    for predicate, desc in SEMANTIC_HINTS:
        if predicate(upper):
            return desc
    return 'business meaning requires SME confirmation'


# --- DESCRIPTION GENERATION ---

def generate_column_descriptions(profiles: Dict[str, Dict[str, Any]], schema_name: str, conn=None) -> Dict[str, Dict[str, str]]:
    """
    Generate semantic descriptions for all columns using LLM.
    
    Returns: {table: {column: "semantic description"}}
    """
    close_conn = False
    if conn is None:
        conn = get_snowflake_connection()
        close_conn = True
    
    descriptions = {}
    
    try:
        cursor = conn.cursor()
        
        for table, columns in profiles.items():
            if not isinstance(columns, dict):
                continue
            
            descriptions[table] = {}
            
            # Use small batches for descriptions to avoid context window / parsing issues
            col_names = list(columns.keys())
            batch_size = 10  # Smaller batch for more reliable parsing
            
            for i in range(0, len(col_names), batch_size):
                batch_cols = col_names[i:i + batch_size]
                col_info_list = []
                
                for col in batch_cols:
                    profile = columns.get(col, {})
                    if not isinstance(profile, dict):
                        profile = {}
                    
                    dtype = profile.get("datatype", "UNKNOWN")
                    
                    # Get sample values safely
                    sample = ""
                    top_vals = profile.get("top_values", [])[:3]
                    if top_vals:
                        samples = []
                        for v in top_vals:
                            val = v.get("value", v) if isinstance(v, dict) else v
                            # Clean and truncate sample value
                            clean_val = str(val).replace("'", "").replace('"', '').replace('\n', ' ')[:25]
                            samples.append(clean_val)
                        sample = ", ".join(samples)
                    
                    hint = infer_semantic(col)
                    col_info_list.append(f"  - {col} ({dtype}): hint='{hint}', samples=[{sample}]")
                
                # Use a clear multi-line prompt with explicit instructions
                prompt = f"""Generate a SHORT business description (under 60 chars) for each column.

Table: {schema_name}.{table}
Columns:
{chr(10).join(col_info_list)}

Return ONLY valid JSON with column names as keys and descriptions as values.
Example: {{"COLUMN_A": "Primary customer ID", "COLUMN_B": "Transaction date"}}

JSON:"""

                
                try:
                    escaped = prompt.replace("'", "''")
                    query = f"SELECT AI_COMPLETE('{LLM_MODEL}', '{escaped}') AS response"
                    cursor.execute(query)
                    raw = cursor.fetchone()[0]
                    
                    # Robust JSON extraction
                    def extract_and_parse_json(text):
                        if not text:
                            return None
                        
                        # Try parsing as-is first
                        try:
                            data = json.loads(text.strip())
                            if isinstance(data, dict):
                                return data
                            if isinstance(data, str):
                                # Recursive call for double-stringified
                                return extract_and_parse_json(data)
                        except:
                            pass
                            
                        # If that failed, try to find { ... }
                        try:
                            start = text.find('{')
                            end = text.rfind('}')
                            if start != -1 and end != -1:
                                cleaned = text[start:end+1]
                                data = json.loads(cleaned)
                                if isinstance(data, dict):
                                    return data
                                if isinstance(data, str):
                                    return extract_and_parse_json(data)
                        except:
                            pass
                        return None

                    batch_results = extract_and_parse_json(raw)
                    
                    if not batch_results or not isinstance(batch_results, dict):
                        print(f"[iterative_mapper] Warning: Could not parse JSON dict from response for {table}")
                        print(f"[iterative_mapper] Raw response snippet: {raw[:200]}")
                        batch_results = {}

                    
                    # Match results back to uppercase column names
                    col_map = {c.upper(): c for c in batch_cols}
                    matched_count = 0
                    for res_col, res_desc in batch_results.items():

                        upper_res = res_col.upper()
                        if upper_res in col_map:
                            descriptions[table][col_map[upper_res]] = res_desc
                            matched_count += 1
                        else:
                            # Try fuzzy match if key differs slightly
                            for actual_upper, actual_col in col_map.items():
                                if actual_upper in upper_res or upper_res in actual_upper:
                                    descriptions[table][actual_col] = res_desc
                                    matched_count += 1
                                    break
                    
                    # If no matches from LLM, use fallback for this batch
                    if matched_count == 0:
                        print(f"[iterative_mapper] No matches from LLM for {table} batch [{i}-{i+batch_size}], using fallback")
                        for col in batch_cols:
                            col_words = col.replace("_", " ").title()
                            descriptions[table][col] = col_words
                    
                except Exception as e:
                    print(f"[iterative_mapper] Description batch failed for {table} [{i}-{i+batch_size}]: {e}")
                    # Fallback for this batch
                    for col in batch_cols:
                        col_words = col.replace("_", " ").title()
                        descriptions[table][col] = col_words
            
            print(f"[iterative_mapper] Generated descriptions for {table}: {len(descriptions.get(table, {}))} columns")
            
        cursor.close()
        
    finally:
        if close_conn and conn:
            conn.close()
    
    return descriptions


def compute_embedding_similarity(source_descriptions: Dict, target_descriptions: Dict, conn=None) -> Dict[str, List[tuple]]:
    """
    Compute embedding similarity between source and target columns.
    
    Returns: {target_table.target_col: [(source_col, source_table, similarity_score), ...]}
    """
    close_conn = False
    if conn is None:
        conn = get_snowflake_connection()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        # Build text lists
        source_texts = []  # (table, col, text)
        target_texts = []  # (table, col, text)
        
        for table, cols in source_descriptions.items():
            for col, desc in cols.items():
                # Use table name in embedding for context
                source_texts.append((table, col, f"Table {table}: Column {col} represents {desc}"))
        
        for table, cols in target_descriptions.items():
            for col, desc in cols.items():
                target_texts.append((table, col, f"Table {table}: Column {col} represents {desc}"))
        
        if not source_texts or not target_texts:
            return {}
        
        # Get embeddings using AI_EMBED (use project standard model)
        EMBED_MODEL = "e5-base-v2"
        
        def get_embeddings(texts):
            embeddings = []
            # Texts is a list of (table, col, text)
            full_texts = [t[2] for t in texts]
            
            for i in range(0, len(full_texts), 20):
                batch = full_texts[i:i+20]
                queries = []
                for idx, text in enumerate(batch):
                    escaped = text.replace("'", "''")[:1000] # Limit length
                    queries.append(f"SELECT {idx} AS idx, AI_EMBED('{EMBED_MODEL}', '{escaped}') AS emb")
                sql = " UNION ALL ".join(queries) + " ORDER BY idx"
                cursor.execute(sql)
                for row in cursor.fetchall():
                    emb = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                    embeddings.append(emb)
            return embeddings
        
        print(f"[iterative_mapper] Generating embeddings for {len(source_texts)} source, {len(target_texts)} target columns...")
        
        source_embeds = get_embeddings(source_texts)
        target_embeds = get_embeddings(target_texts)
        
        # Compute cosine similarity
        import numpy as np
        source_arr = np.array(source_embeds)
        target_arr = np.array(target_embeds)
        
        # Normalize
        source_norm = source_arr / np.linalg.norm(source_arr, axis=1, keepdims=True)
        target_norm = target_arr / np.linalg.norm(target_arr, axis=1, keepdims=True)
        
        # Similarity matrix: target x source
        similarity = np.dot(target_norm, source_norm.T)
        
        # For each target, get top 15 similar sources
        results = {}
        for t_idx, (t_table, t_col, _) in enumerate(target_texts):
            key = f"{t_table}.{t_col}"
            scores = similarity[t_idx]
            top_indices = np.argsort(scores)[::-1][:15]
            
            results[key] = []
            for s_idx in top_indices:
                s_table, s_col, _ = source_texts[s_idx]
                score = float(scores[s_idx])
                results[key].append((s_col, s_table, score))
        
        print(f"[iterative_mapper] Computed similarity matrix: {len(results)} targets")
        cursor.close()
        return results

        
    except Exception as e:
        print(f"[iterative_mapper] Embedding similarity failed: {e}")
        traceback.print_exc()
        return {}
    finally:
        if close_conn and conn:
            conn.close()


def convert_profiles_to_schema(profiles: Dict[str, Dict[str, Any]], schema_name: str, descriptions: Dict[str, Dict[str, str]] = None) -> List[Dict]:
    """
    Convert profile dict format to new schema JSON format.
    
    Input:  { table: { col: profile_dict } }
    Output: [{ table_name, schema_name, columns: [...] }]
    
    Args:
        profiles: Profile data
        schema_name: Name of the schema
        descriptions: Optional pre-generated semantic descriptions {table: {col: "description"}}
    """
    result = []
    for table, columns in profiles.items():
        if not isinstance(columns, dict):
            continue
        
        table_entry = {
            "table_name": table,
            "schema_name": schema_name,
            "description": "",
            "columns": []
        }

        
        for col, profile in columns.items():
            if not isinstance(profile, dict):
                # If profile is not a dict, create minimal entry
                table_entry["columns"].append({
                    "column_name": col,
                    "column_type": "UNKNOWN",
                    "sample_data": "",
                    "nullable": True,
                    "primary_key": False,
                    "description": f"Column {col}"
                })
                continue
            
            # Extract sample data from top_values
            sample_parts = []
            top_vals = profile.get("top_values", [])
            if isinstance(top_vals, list):
                for v in top_vals[:3]:
                    if isinstance(v, dict):
                        sample_parts.append(str(v.get("value", ""))[:20])
                    else:
                        sample_parts.append(str(v)[:20])
            sample_data = ", ".join(sample_parts)
            
            # Check if nullable
            nulls = profile.get("nulls", 0)
            nullable = nulls > 0
            
            # Detect potential primary key from naming
            is_pk = any(pk in col.upper() for pk in ["_ID", "_KEY", "_PK", "ID_"])
            
            # Get data type
            dtype = profile.get("datatype", "UNKNOWN")
            
            # Use pre-generated semantic description if available
            col_description = ""
            if descriptions and table in descriptions and col in descriptions[table]:
                col_description = descriptions[table][col]
            
            # Fallback to simple description from column name
            if not col_description:
                col_words = col.replace("_", " ").replace("-", " ").title()
                col_description = f"{col_words}"
            
            table_entry["columns"].append({
                "column_name": col,
                "column_type": dtype,
                "sample_data": sample_data,
                "nullable": nullable,
                "primary_key": is_pk,
                "description": col_description
            })


        
        result.append(table_entry)
    
    return result



def generate_mapping_prompt(target_batch: List[Dict], source_chunk: List[Dict], 
                           current_state: Optional[Dict], business_context: str = "") -> str:
    """Generate the LLM prompt for mapping."""
    return f"""### ROLE
You are a Senior Data Engineer. Map target columns to source columns with high precision.

### INPUT DATA
- TARGET BATCH: {json.dumps(target_batch, indent=2)}
- SOURCE CHUNK: {json.dumps(source_chunk, indent=2)}
- PREVIOUS PROGRESS: {json.dumps(current_state, indent=2) if current_state else "Start of mapping."}
- BUSINESS CONTEXT: {business_context if business_context else "No specific context provided."}

### OUTPUT FIELD DEFINITIONS (How to fill the JSON):
1. 'target_column_name': The name of the target column currently being processed.
2. 'target_description': A SHORT semantic description of what this target column represents (e.g., "Primary key for customer dimension", "Customer's full legal name").
3. 'source_column_name' (Array): An array of ALL matching source columns from ANY source table. Each entry contains:
   - 'source_column_name': The exact name of the matching column in the source.
   - 'source_table_name': The table where the source column resides.
   - 'source_schema_name': The schema where the source table resides.
   - 'source_description': A SHORT semantic description of what this source column represents (e.g., "Unique customer identifier from CRM system").
   - 'mapping_score': A confidence score from 0-100 indicating how confident you are in this mapping (100 = perfect match, 0 = no match).
   - 'justification': A brief technical reason for the match.
4. 'final_transformation_logic': 
   - Provide a Snowflake SQL expression. 
   - If a direct match: "SOURCE_COLUMN"
   - If a derivation: e.g., "TRIM(UPPER(FIRST_NAME)) || ' ' || LAST_NAME"
   - If no match found yet: Return an empty string "".
5. 'work_notes': 
   - Acts as a memory scratchpad. 
   - Describe partial matches (e.g., "Found 1 of 2 columns for a composite key"). 
   - List missing pieces required to finalize 'final_transformation_logic'.

### LOGIC RULES:
- CRITICAL: If you find a matching source column in THIS CHUNK, ADD IT to the 'source_column_name' array.
- KEEP ALL EXISTING SOURCES from 'PREVIOUS PROGRESS' - do NOT remove sources from other tables.
- A target may have MULTIPLE valid source mappings from DIFFERENT tables. Include ALL of them.
- Only replace a source mapping if the new one is from the SAME table but is a better match.
- Return a valid JSON object matching the schema.
- IMPORTANT: Always provide source_description and mapping_score for each mapping.
"""





def call_cortex_structured(prompt: str, conn=None) -> Optional[Dict]:
    """Call Snowflake Cortex and parse JSON response."""
    close_conn = False
    if conn is None:
        conn = get_snowflake_connection()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        try:
            # Append JSON instruction to prompt
            full_prompt = prompt + """

IMPORTANT: Return ONLY a valid JSON object with no additional text, markdown, or code blocks.
The JSON must have this exact structure:
{
  "mapped_columns": [
    {
      "target_column_name": "string",
      "source_column_name": [
        {"source_column_name": "string", "source_table_name": "string", "source_schema_name": "string", "justification": "string"}
      ],
      "final_transformation_logic": "string",
      "work_notes": "string"
    }
  ]
}
"""
            
            # Escape single quotes for SQL
            escaped_prompt = full_prompt.replace("'", "''")
            
            # Use simple AI_COMPLETE call
            query = f"SELECT AI_COMPLETE('{LLM_MODEL}', '{escaped_prompt}') AS response"
            cursor.execute(query)
            raw_response = cursor.fetchone()[0]
            
            print(f"[iterative_mapper] Cortex returned {len(raw_response)} chars")
            
            # Extract JSON block robustly
            content = raw_response.strip()
            
            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```', 2)[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            # Find JSON boundaries
            start = content.find('{')
            end = content.rfind('}')
            
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            # CRITICAL FIX: Handle escaped characters from Cortex
            # Cortex sometimes returns JSON with escaped quotes and newlines
            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            # MOST IMPORTANT: Unescape quotes - Cortex returns \" instead of "
            content = content.replace('\\"', '"')
            
            # Try to parse JSON - handle multiple formats
            result = None
            try:
                result = json.loads(content)
                
                # Handle double-stringified JSON
                parse_attempts = 0
                while isinstance(result, str) and parse_attempts < 3:
                    result = json.loads(result)
                    parse_attempts += 1
                    
            except json.JSONDecodeError as e:
                print(f"[iterative_mapper] JSON parse error: {e}")
                # Try one more time with aggressive cleaning
                try:
                    # Remove all control characters except newlines, tabs, carriage returns
                    import re
                    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
                    result = json.loads(cleaned)
                    print(f"[iterative_mapper] Successfully parsed after cleaning control characters")
                except:
                    # Log a larger snippet for debugging if it fails
                    print(f"[iterative_mapper] Content length: {len(content)}, First 300 chars: {content[:300]}")
                    print(f"[iterative_mapper] Last 200 chars: {content[-200:] if len(content) > 200 else 'N/A'}")
                    # Show hex of first few characters for debugging
                    hex_start = ' '.join(f'{ord(c):02x}' for c in content[:20])
                    print(f"[iterative_mapper] First 20 chars hex: {hex_start}")
                    return None
            
            # Validate structure
            if isinstance(result, dict) and "mapped_columns" in result:
                return result
            else:
                print(f"[iterative_mapper] Invalid response structure: {type(result)}")
                return None

            
        except Exception as e:
            print(f"[iterative_mapper] Cortex call error: {e}")
            traceback.print_exc()
            return None
        finally:
            cursor.close()
    finally:
        if close_conn and conn:
            conn.close()


def run_iterative_mapping(
    source_profiles: Dict[str, Dict[str, Any]],
    target_profiles: Dict[str, Dict[str, Any]],
    source_schema_name: str = "BRONZE",
    target_schema_name: str = "SILVER",
    business_context: str = "",
    progress_callback=None,
    source_descriptions: Dict[str, Dict[str, str]] = None,
    target_descriptions: Dict[str, Dict[str, str]] = None,
    similarity_hints: Dict[str, List[tuple]] = None
) -> List[Dict[str, Any]]:
    """
    Run iterative batch-based mapping from source to target.
    
    Args:
        source_profiles: Bronze profiles {table: {col: profile}}
        target_profiles: Silver profiles {table: {col: profile}}
        source_schema_name: Name of the source schema
        target_schema_name: Name of the target schema
        business_context: User-provided mapping hints
        progress_callback: Optional callback for progress updates (current, total, message)
        source_descriptions: Pre-generated semantic descriptions for source columns
        target_descriptions: Pre-generated semantic descriptions for target columns
        similarity_hints: Embedding similarity results {target: [(source, table, score), ...]}

    
    Returns:
        List of mapping dictionaries
    """
    print(f"[iterative_mapper] === STARTING ITERATIVE MAPPING ===")
    
    # Convert to new schema format - use pre-generated descriptions if available
    source_data = convert_profiles_to_schema(source_profiles, source_schema_name, source_descriptions)
    target_data = convert_profiles_to_schema(target_profiles, target_schema_name, target_descriptions)

    
    print(f"[iterative_mapper] Source: {len(source_data)} tables")
    print(f"[iterative_mapper] Target: {len(target_data)} tables")
    
    all_results = []
    total_target_cols = sum(len(t.get("columns", [])) for t in target_data)
    processed_cols = 0
    
    # Get connection once for all calls
    conn = get_snowflake_connection()
    
    try:
        for target_table in target_data:
            t_table_name = target_table.get("table_name", "UNKNOWN")
            t_cols = target_table.get("columns", [])
            
            print(f"[iterative_mapper] Processing target table: {t_table_name} ({len(t_cols)} columns)")
            
            # Process target columns in batches
            for i in range(0, len(t_cols), TARGET_BATCH_SIZE):
                target_batch = t_cols[i:i + TARGET_BATCH_SIZE]
                
                # Add target table context to each column
                for col in target_batch:
                    col["target_table_name"] = t_table_name
                    col["target_schema_name"] = target_schema_name
                
                # Initialize state for this batch
                current_state = {
                    "mapped_columns": [
                        {
                            "target_column_name": c["column_name"],
                            "source_column_name": [],
                            "final_transformation_logic": "",
                            "work_notes": "Awaiting initial scan."
                        } for c in target_batch
                    ]
                }
                
                # Scan all source tables
                for source_table in source_data:
                    s_table_name = source_table.get("table_name", "UNKNOWN")
                    s_cols = source_table.get("columns", [])
                    
                    # Process source columns in chunks
                    for j in range(0, len(s_cols), SOURCE_CHUNK_SIZE):
                        source_chunk = s_cols[j:j + SOURCE_CHUNK_SIZE]
                        
                        # Add source table context to each column
                        for col in source_chunk:
                            col["source_table_name"] = s_table_name
                            col["source_schema_name"] = source_schema_name
                        
                        # Generate prompt and call LLM
                        prompt = generate_mapping_prompt(
                            target_batch, 
                            source_chunk, 
                            current_state,
                            business_context
                        )
                        
                        new_state = call_cortex_structured(prompt, conn)
                        
                        if new_state and "mapped_columns" in new_state:
                            current_state = new_state
                            print(f"[iterative_mapper] Updated state from {s_table_name}")
                        else:
                            print(f"[iterative_mapper] No valid response for {s_table_name} chunk")
                
                # Add completed mappings to results
                for mapping in current_state.get("mapped_columns", []):
                    # Get target column description from target_batch
                    target_col_name = mapping.get("target_column_name", "")
                    target_desc = ""
                    target_dtype = ""
                    for tc in target_batch:
                        if tc.get("column_name") == target_col_name:
                            target_desc = tc.get("description", "")
                            target_dtype = tc.get("column_type", "")
                            break
                    
                    # Convert to flat output format for CSV
                    # Get target description from LLM response (semantic), fallback to profile
                    target_desc_llm = mapping.get("target_description", "")
                    
                    base_mapping = {
                        "TargetTable": t_table_name,
                        "TargetSchema": target_schema_name,
                        "TargetColumn": target_col_name,
                        "TargetDataType": target_dtype,
                        "TargetDescription": target_desc_llm if target_desc_llm else target_desc,
                        "TransformationLogic": mapping.get("final_transformation_logic", ""),
                        "WorkNotes": mapping.get("work_notes", "")
                    }
                    
                    sources = mapping.get("source_column_name", [])
                    if sources:
                        # Deduplicate sources based on table and column name
                        seen_sources = set()
                        for src in sources:
                            src_table = src.get("source_table_name", "UNKNOWN")
                            src_col = src.get("source_column_name", "UNKNOWN")
                            source_key = f"{src_table}.{src_col}"
                            
                            if source_key in seen_sources or src_col == "UNMAPPED":
                                continue
                            seen_sources.add(source_key)
                            
                            # Get LLM-generated semantic description and score
                            source_desc_llm = src.get("source_description", "")
                            mapping_score = src.get("mapping_score", 0)
                            
                            # Fallback: Lookup source data type from source_data
                            source_dtype = ""
                            for s_tbl in source_data:
                                if s_tbl.get("table_name") == src_table:
                                    for s_col in s_tbl.get("columns", []):
                                        if s_col.get("column_name") == src_col:
                                            source_dtype = s_col.get("column_type", "")
                                            break
                                    break
                            
                            all_results.append({
                                **base_mapping,
                                "SourceTable": src_table,
                                "SourceSchema": src.get("source_schema_name", ""),
                                "SourceColumn": src_col,
                                "SourceDataType": source_dtype,
                                "SourceDescription": source_desc_llm,
                                "MappingScore": mapping_score,
                                "Justification": src.get("justification", "")
                            })
                    
                    # If after deduplication we have no valid sources, add one N/A row
                    if not any(r["TargetColumn"] == target_col_name and r["SourceTable"] != "N/A" for r in all_results if r["TargetColumn"] == target_col_name):
                        # check if we already added a row for this target_col_name
                        added_for_this = [r for r in all_results if r["TargetColumn"] == target_col_name and r["TargetTable"] == t_table_name]
                        if not added_for_this:
                            all_results.append({
                                **base_mapping,
                                "SourceTable": "N/A",
                                "SourceSchema": "N/A",
                                "SourceColumn": "N/A",
                                "SourceDataType": "N/A",
                                "SourceDescription": "No suitable source column found in current scan.",
                                "MappingScore": 0,
                                "Justification": "No matching source column found"
                            })



                
                processed_cols += len(target_batch)
                
                if progress_callback:
                    progress_callback(processed_cols, total_target_cols, f"Mapped {processed_cols}/{total_target_cols} columns")
    
    except Exception as e:
        print(f"[iterative_mapper] Error: {e}")
        traceback.print_exc()
    finally:
        conn.close()
    
    print(f"[iterative_mapper] === COMPLETED: {len(all_results)} mappings ===")
    return all_results


if __name__ == "__main__":
    print("Iterative Mapper module loaded successfully.")
    print(f"Config: TARGET_BATCH={TARGET_BATCH_SIZE}, SOURCE_CHUNK={SOURCE_CHUNK_SIZE}, MODEL={LLM_MODEL}")
