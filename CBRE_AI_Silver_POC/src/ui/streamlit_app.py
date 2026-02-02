import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to pythonpath so we can import our modules
base_dir = Path(__file__).resolve().parents[2]
src_dir = base_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from extract.profile_real_data import list_tables, profile_table, get_connection

# ... imports remain ...

# Load env
load_dotenv()

st.set_page_config(page_title="Silver Model Builder", layout="wide")
st.title("Silver Model Builder: Control Center")

# --- CACHED UTILS ---
# We wrap the imported functions to cache their results.
# 'ttl=600' means stats refresh every 10 mins, preventing stale schemas.

@st.cache_data(ttl=600, show_spinner=False)
def list_tables_cached(account, user, role, warehouse, database, schema):
    # We pass connection params as args so cache invalidates if they change
    os.environ["SNOWFLAKE_ACCOUNT"] = account or ""
    os.environ["SNOWFLAKE_USER"] = user or ""
    os.environ["SNOWFLAKE_ROLE"] = role or ""
    os.environ["SNOWFLAKE_WAREHOUSE"] = warehouse or ""
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    os.environ["SNOWFLAKE_SCHEMA"] = schema or ""
    
    conn = get_connection()
    try:
        return list_tables(conn, limit=200)
    finally:
        conn.close()

@st.cache_data(ttl=600, show_spinner=False)
def get_column_info_cached(account, user, role, warehouse, database, schema, table_name):
    # Re-set env for safety in thread-local execution
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    os.environ["SNOWFLAKE_SCHEMA"] = schema or ""
    
    conn = get_connection()
    try:
        return get_column_info(conn, table_name)
    finally:
        conn.close()

@st.cache_data(ttl=3600, show_spinner=False)
def profile_table_cached(account, user, role, warehouse, database, schema, table_name, target_columns):
    """Cached wrapper for profiling a specific table."""
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    os.environ["SNOWFLAKE_SCHEMA"] = schema or ""
    conn = get_connection()
    try:
        return profile_table(conn, table_name, target_columns=target_columns)
    finally:
        conn.close()

# --- CACHED AI WRAPPERS ---

@st.cache_data(show_spinner=False)
def generate_descriptions_cached(metadata_struct, all_profiles):
    """Cached wrapper for description generation."""
    # Create unique temp paths for this cache entry execution
    import uuid
    run_id = str(uuid.uuid4())[:8]
    temp_dir = base_dir / "data" / "real_run" / f"temp_cache_{run_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    prof_path = temp_dir / "profile.json"
    meta_path = temp_dir / "metadata.json"
    desc_path = temp_dir / "descriptions.json"
    
    try:
        prof_path.write_text(json.dumps(all_profiles), encoding='utf-8')
        meta_path.write_text(json.dumps(metadata_struct), encoding='utf-8')
        
        generate_descriptions('llm', meta_path, prof_path, desc_path)
        
        return json.loads(desc_path.read_text(encoding='utf-8'))
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@st.cache_data(show_spinner=False)
def get_embeddings_cached(texts):
    return get_embeddings_snowflake(texts)

@st.cache_data(show_spinner=False)
def generate_silver_model_cached(cluster_payload):
    return generate_silver_model(cluster_payload)

# --- SIDEBAR: CONNECTION ---
st.sidebar.header("🔌 Snowflake Connection")

# Allow user to override env vars
account = st.sidebar.text_input("Account", value=os.getenv("SNOWFLAKE_ACCOUNT", ""))
user = st.sidebar.text_input("User", value=os.getenv("SNOWFLAKE_USER", ""))
role = st.sidebar.text_input("Role", value=os.getenv("SNOWFLAKE_ROLE", ""))
warehouse = st.sidebar.text_input("Warehouse", value=os.getenv("SNOWFLAKE_WAREHOUSE", ""))
database = st.sidebar.text_input("Database", value=os.getenv("SNOWFLAKE_DATABASE", ""))
schema = st.sidebar.text_input("Schema", value=os.getenv("SNOWFLAKE_SCHEMA", "SILVER"))

if st.sidebar.button("Test Connection"):
    try:
        # Temporarily set env vars for the get_connection utility
        os.environ["SNOWFLAKE_ACCOUNT"] = account
        os.environ["SNOWFLAKE_USER"] = user
        os.environ["SNOWFLAKE_ROLE"] = role
        os.environ["SNOWFLAKE_WAREHOUSE"] = warehouse
        os.environ["SNOWFLAKE_DATABASE"] = database
        os.environ["SNOWFLAKE_SCHEMA"] = schema
        
        conn = get_connection()
        st.sidebar.success("✅ Connected!")
        conn.close()
    except Exception as e:
        st.sidebar.error(f"❌ Connection Failed: {e}")


# --- MAIN PANEL ---

import json
from extract.profile_real_data import list_tables, profile_table, get_connection, get_column_info
from profiling.generate_long_descriptions import generate_descriptions
from ai.semantic_clustering import build_texts, get_embeddings_snowflake, cluster, fallback_embeddings
from ai.silver_model_generator import generate_silver_model
from ai.generate_silver_snowflake_ddl import parse_robust, sanitize, build_column_line, DEFAULT_PK_CANDIDATES
# ... (Connection code remains same, omitted for brevity in replacement if possible, but here we replace whole file context usually so let's keep it safe or assume partial replacement if I knew line numbers perfectly. Assuming I replace the whole MAIN PANEL logic onwards)

# ... (Previous sidebar code ends around line 48)

# --- MAIN PANEL ---

st.subheader("1️⃣ Select Bronze Tables")

# Fetch tables (Cached)
tables = []
try:
    tables = list_tables_cached(account, user, role, warehouse, database, schema)
except Exception:
    st.info("Please configure connection in sidebar to see tables.")

selected_tables = st.multiselect("Choose tables:", tables)

# --- COLUMN SELECTION ---
selected_columns_map = {} # {table: [col1, col2]}

if selected_tables:
    st.subheader("2️⃣ Select Target Columns")
    
    # get_column_info is now cached, so multiple iterations are fast
    for table in selected_tables:
        # distinct key for each expander to avoid conflicts
        with st.expander(f"Settings for `{table}`", expanded=False):
            # Pass all connection params to ensure cache key matches current sidebar state
            cols_info = get_column_info_cached(account, user, role, warehouse, database, schema, table)
            col_names = [c['name'] for c in cols_info]
            
            # Create a dataframe for the editor
            df_cols = pd.DataFrame({
                "Select": [True] * len(col_names),
                "Column Name": col_names
            })
            
            # scalable data editor with search/checkboxes
            edited_df = st.data_editor(
                df_cols,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Include",
                        help="Check to include in profiling",
                        default=True,
                    ),
                    "Column Name": st.column_config.TextColumn(
                        "Column",
                        width="medium",
                        disabled=True
                    )
                },
                hide_index=True,
                use_container_width=True,
                key=f"editor_{table}"
            )
            
            # Filter back to list
            picked = edited_df[edited_df["Select"]]["Column Name"].tolist()
            selected_columns_map[table] = picked

st.write("---")
st.subheader("3️⃣ Pipeline Execution")

if st.button("🚀 Run End-to-End Pipeline"):
    if not selected_tables:
        st.warning("Select specific tables/columns first.")
    else:
        monitor_placeholder = st.empty()
        progress = st.progress(0)
        
        pipeline_steps = [
            "1. Profiling & Metadata Extraction",
            "2. AI Description Generation",
            "3. Semantic Clustering",
            "4. Silver Model Logic Design",
            "5. Snowflake DDL Emission"
        ]

        def update_monitor(active_idx):
            with monitor_placeholder.container(border=True):
                st.write("### 🛤️ Pipeline Process Monitor")
                cols = st.columns(len(pipeline_steps))
                for i, name in enumerate(pipeline_steps):
                    with cols[i]:
                        if i < active_idx:
                            st.markdown(f"✅ **{name}**\n\nDone")
                        elif i == active_idx:
                            st.markdown(f"⏳ **{name}**\n\nRunning...")
                        else:
                            st.markdown(f"⚪ **{name}**\n\nPending")
                if active_idx < len(pipeline_steps):
                    st.write(f"Current Activity: **{pipeline_steps[active_idx]}**")
                else:
                    st.write("Current Activity: **🏁 Pipeline Complete**")

        try:
            # 1. PROFILING
            update_monitor(0)
            conn = get_connection()
            all_profiles = {}
            
            for i, table in enumerate(selected_tables):
                target_cols = selected_columns_map[table]
                if not target_cols:
                    continue
                    
                prof = profile_table_cached(account, user, role, warehouse, database, schema, table, tuple(target_cols))
                all_profiles[table] = prof
                progress.progress((i + 1) / len(selected_tables) * 0.2)
            
            conn.close()
            
            # Convert to "Metadata" format required by description generator
            # {Table: {columns: [{name, datatype}]}}
            metadata_struct = {}
            for t, p in all_profiles.items():
                metadata_struct[t] = {"columns": [{"name": k, "datatype": v.get("datatype")} for k, v in p.items()]}
            
            # 2. DESCRIPTIONS
            update_monitor(1)
            progress.progress(0.4)
            
            # Call cached generator
            desc_data = generate_descriptions_cached(metadata_struct, all_profiles)
            
            # 3. CLUSTERING
            update_monitor(2)
            progress.progress(0.6)
            
            texts = build_texts(desc_data) # Returns ["Table.Col: Desc", ...]
            labels = [t.split(':', 1)[0] for t in texts]
            
            if not texts:
                 st.error("No descriptions generated. Check headers/profiling.")
                 st.stop()

            # Embed
            try:
                # Assuming EMBED_MODEL is set in env
                embeds = get_embeddings_cached(texts)
            except Exception as e:
                st.warning(f"Snowflake Embed failed ({e}), falling back to TF-IDF")
                embeds = fallback_embeddings(texts)
                
            clusters = cluster(embeds, labels, k=max(2, int(len(texts)**0.5)))
            
            cluster_payload = {'embed_model': 'ui-run', 'cluster_count': len(clusters), 'clusters': clusters}
            
            # 4. SILVER MODEL
            update_monitor(3)
            progress.progress(0.8)
            
            model_json_str = generate_silver_model_cached(cluster_payload) # Returns string
            model_data = parse_robust(model_json_str)
            
            # Handle potential double-encoding (LLM sometimes returns stringified JSON)
            if isinstance(model_data, str):
                model_data = parse_robust(model_data)

            if not isinstance(model_data, dict) or "entities" not in model_data:
                st.error("Failed to generate valid model JSON from LLM.")
                st.code(model_json_str)
                st.stop()
                
            # 5. DDL GENERATION
            update_monitor(4)
            progress.progress(0.9)
            
            ddl_statements = []
            csv_rows = []
            target_schema = "SILVER" # Force target schema
            
            # Reuse logic from generate_silver_snowflake_ddl.py
            for entity in model_data.get("entities", []):
                t_name = sanitize(entity.get("entity_name"))
                attrs_sql = []
                pk_cols = []
                
                for attr in entity.get("attributes", []):
                    col = sanitize(attr.get("name"))
                    dtype = attr.get("datatype")
                    desc = attr.get("description", "")
                    desc_safe = desc.replace("'", "''")
                    attrs_sql.append(f"    {col} {dtype} COMMENT '{desc_safe}'")
                    
                    if attr.get("is_pk") or col in DEFAULT_PK_CANDIDATES:
                        pk_cols.append(col)
                    
                    # Collect CSV data (Mapping Format)
                    src_cols = attr.get("source_columns", [])
                    if not src_cols:
                        src_cols = ["UNKNOWN.UNKNOWN.UNKNOWN"]
                        
                    for src in src_cols:
                        # Attempt to parse SCHEMA.TABLE.COL
                        parts = src.split('.')
                        if len(parts) == 3:
                            s_schema, s_table, s_col = parts
                        elif len(parts) == 2:
                             s_schema, s_table, s_col = schema, parts[0], parts[1]
                        else:
                             s_schema, s_table, s_col = schema, "UNKNOWN", parts[0]
                        
                        # Get source metadata from all_profiles
                        s_desc = ""
                        if s_table in desc_data and s_col in desc_data[s_table]:
                            s_desc = desc_data[s_table][s_col]
                        
                        s_type = ""
                        if s_table in all_profiles and s_col in all_profiles[s_table]:
                            s_type = all_profiles[s_table][s_col].get("datatype", "")

                        csv_rows.append({
                            "SourceSchema": s_schema,
                            "SourceTableName": s_table,
                            "SourceColumn": s_col,
                            "SourceDescription": s_desc,
                            "SourceDataType": s_type,
                            "TargetSchema": target_schema,
                            "TargetTableName": t_name,
                            "TargetColumn": col,
                            "TargetDataType": dtype,
                            "TargetDescription": desc,
                            "TransformationRule": "Direct Map" if s_col == col else "Renamed/Transformed"
                        })
                
                pk_clause = f",\n    PRIMARY KEY ({', '.join(pk_cols)})" if pk_cols else ""
                full_name = f"{database}.{target_schema}.{t_name}"
                ddl = f"CREATE TRANSIENT TABLE IF NOT EXISTS {full_name} (\n" + ",\n".join(attrs_sql) + pk_clause + "\n);"
                ddl_statements.append(ddl)
            
            final_sql = "\n\n".join(ddl_statements)
            progress.progress(1.0)
            update_monitor(5) # All Done
            st.success("✅ Silver Model Pipeline: Cycle Complete!")
            
            # --- RESULTS ---
            st.subheader("4️⃣ Generated Artifacts")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 DDL Script", "🧠 Logical Model (JSON)", "📊 Profile Data", "📥 Model Export", "🔗 Lineage"])
            
            with tab1:
                st.code(final_sql, language="sql")
                st.download_button("Download DDL", final_sql, "silver_model.sql", "text/plain")
                
            with tab2:
                st.json(model_data)
                
            with tab3:
                # Flatten profile for grid
                stats_rows = []
                for t, p in all_profiles.items():
                    for c, s in p.items():
                        s['Table'] = t
                        s['Column'] = c
                        stats_rows.append(s)
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
            
            with tab4:
                if csv_rows:
                    df_export = pd.DataFrame(csv_rows)
                    st.dataframe(df_export, use_container_width=True)
                    
                    csv = df_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Model Definition (CSV)",
                        data=csv,
                        file_name="silver_model_definition.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No model definition data available for export.")

            with tab5:
                # Generate Mermaid Lineage
                unique_links = set((r['SourceTableName'], r['TargetTableName']) for r in csv_rows)
                if unique_links:
                    mermaid_lines = ["graph LR"]
                    mermaid_lines.append("  subgraph Bronze")
                    for s_table in sorted(set(r[0] for r in unique_links)):
                        # Sanitize table names for Mermaid IDs (replace . or spaces)
                        s_id = s_table.replace(".", "_").replace(" ", "_")
                        mermaid_lines.append(f"    B_{s_id}[\"{s_table}\"]")
                    mermaid_lines.append("  end")
                    mermaid_lines.append("  subgraph Silver")
                    for t_table in sorted(set(r[1] for r in unique_links)):
                        t_id = t_table.replace(".", "_").replace(" ", "_")
                        mermaid_lines.append(f"    S_{t_id}[\"{t_table}\"]")
                    mermaid_lines.append("  end")
                    for s_table, t_table in sorted(unique_links):
                        s_id = s_table.replace(".", "_").replace(" ", "_")
                        t_id = t_table.replace(".", "_").replace(" ", "_")
                        mermaid_lines.append(f"  B_{s_id} --> S_{t_id}")
                    
                    mermaid_code = "\n".join(mermaid_lines)
                    st.markdown(f"### Data Flow: Bronze ➔ Silver")
                    st.markdown(f"```mermaid\n{mermaid_code}\n```")
                else:
                    st.info("No lineage data available.")
                
        except Exception as e:
            st.error(f"Pipeline Failed: {e}")
            st.exception(e)
