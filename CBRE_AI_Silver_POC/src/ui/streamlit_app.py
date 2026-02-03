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

from extract.profile_real_data import list_tables, get_connection

# ... imports remain ...

# Load env
load_dotenv()

from decimal import Decimal
import json
from datetime import datetime, date

class SnowflakeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Snowflake/Decimal/Datetime types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj) if obj % 1 else int(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

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
def profile_table_cached(account, user, role, warehouse, database, schema, table_name, target_columns, sample_pct=100):
    """Cached wrapper for profiling a specific table."""
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    os.environ["SNOWFLAKE_SCHEMA"] = schema or ""
    conn = get_connection()
    try:
        import extract.profile_real_data
        import importlib
        importlib.reload(extract.profile_real_data)
        return extract.profile_real_data.profile_table(conn, table_name, target_columns=target_columns, sample_pct=sample_pct)
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
        prof_path.write_text(json.dumps(all_profiles, cls=SnowflakeEncoder), encoding='utf-8')
        meta_path.write_text(json.dumps(metadata_struct, cls=SnowflakeEncoder), encoding='utf-8')
        
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

# --- SESSION STATE ---
if "profiles" not in st.session_state:
    st.session_state.profiles = None
if "model_data" not in st.session_state:
    st.session_state.model_data = None
if "desc_data" not in st.session_state:
    st.session_state.desc_data = None

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

# --- PHASE 1: PROFILING ---
st.subheader("3️⃣ Profiling")

col1, col2 = st.columns([2, 1])
with col2:
    sample_pct = st.slider("Sample Rate (%)", 1, 100, 100, help="For tables with millions of rows, use a lower sample rate (e.g., 10%) for instant results.")

if st.button("🔍 Profile Data", use_container_width=True):
    if not selected_tables:
        st.warning("Please select at least one table.")
    else:
        with st.status("🛠️ Running Data Quality Analysis...", expanded=True) as status:
            conn = None
            all_profiles = {}
            try:
                conn = get_connection()
                total_cols_profiled = 0
                for table in selected_tables:
                    target_cols = selected_columns_map.get(table, [])
                    if not target_cols: continue
                    st.write(f"Profiling `{table}` ({len(target_cols)} columns, {sample_pct}% sample)...")
                    prof = profile_table_cached(account, user, role, warehouse, database, schema, table, tuple(target_cols), sample_pct)
                    all_profiles[table] = prof
                    total_cols_profiled += len(prof)
                
                st.session_state.profiles = all_profiles
                if total_cols_profiled > 0:
                    status.update(label=f"✅ Profiling Complete! ({total_cols_profiled} columns analyzed)", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ Profiling finished, but no columns were analyzed.", state="complete", expanded=True)
                    st.warning("No columns were analyzed. Ensure you checked the 'Select' box for columns in step 2.")
            except Exception as e:
                st.error(f"Profiling Error: {str(e)}")
                status.update(label="❌ Profiling Failed", state="error", expanded=True)
            finally:
                if conn:
                    conn.close()

if st.session_state.profiles:
    st.write("---")
    st.subheader("📊 Data Quality Grid")
    
    # Flatten all columns for the rich table
    all_col_stats = []
    for table, cols in st.session_state.profiles.items():
        for col_name, stats in cols.items():
            total = stats.get('total', 0)
            nulls = stats.get('nulls', 0)
            phys_nulls = stats.get('physical_nulls', 0)
            valid = total - nulls
            completeness = (valid / total * 100) if total > 0 else 0
            
            # Format range
            rng = stats.get('range', {})
            min_val = rng.get('min', 'N/A')
            max_val = rng.get('max', 'N/A')
            
            # Format top values
            top_vals = stats.get('top_values', [])
            if top_vals and isinstance(top_vals[0], dict):
                top_str = ", ".join([f"{v['value']} ({v['count']})" for v in top_vals])
            else:
                top_str = ", ".join([str(v) for v in top_vals])

            all_col_stats.append({
                "Table": table,
                "Column": col_name,
                "Type": stats.get('datatype', 'UNKNOWN'),
                "Completeness": completeness,
                "Null %": (nulls / total * 100) if total > 0 else 0,
                "Distinct": stats.get('distinct', 0),
                "Min": min_val,
                "Max": max_val,
                "Top Values": top_str
            })
    
    df_dq = pd.DataFrame(all_col_stats)

    if df_dq.empty:
        st.info("No profiling data available. Please select tables and columns, then click 'Profile Data'.")
    else:
        # 1. Summary Metrics Header
        m1, m2, m3, m4, m5 = st.columns(5)
        
        avg_comp = df_dq["Completeness"].mean()
        total_records = sum(stats.get('total', 0) for table_stats in st.session_state.profiles.values() for stats in table_stats.values())
        unique_cells = df_dq["Distinct"].sum()
        
        # Grade logic
        grade = "A" if avg_comp > 90 else "B" if avg_comp > 75 else "C" if avg_comp > 50 else "D"
        grade_color = "green" if grade == "A" else "orange" if grade in ["B", "C"] else "red"

        m1.metric("Overall Health", f"{grade}", help=f"Average Score: {avg_comp:.1f}%")
        m2.metric("Avg Completeness", f"{avg_comp:.1f}%")
        m3.metric("Total Attributes", len(df_dq))
        m4.metric("High Quality (>95%)", len(df_dq[df_dq["Completeness"] > 95]))
        m5.metric("Action Needed (<50%)", len(df_dq[df_dq["Completeness"] < 50]), help="Attributes with more than 50% missing or placeholder data.")

        # Second row of technical metrics
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.caption(f"🚀 Total Rows Analyzed: **{total_records:,}**")
        c2.caption(f"🔑 Total Unique Values: **{unique_cells:,}**")
        
        type_counts = df_dq["Type"].value_counts().to_dict()
        type_str = ", ".join([f"**{k}**: {v}" for k, v in type_counts.items()])
        c3.caption(f"📂 Type Distribution: {type_str}")

        # 2. The Professional DQ Grid
        st.data_editor(
            df_dq,
            column_config={
                "Completeness": st.column_config.ProgressColumn(
                    "Completeness",
                    help="Ratio of non-null, non-placeholder values",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Null %": st.column_config.NumberColumn(
                    "Null %",
                    help="Includes physical nulls, empty strings, and placeholders like 'N/A'",
                    format="%.1f%%"
                ),
                "Top Values": st.column_config.TextColumn(
                    "Top Values",
                    width="large"
                )
            },
            hide_index=True,
            use_container_width=True,
            disabled=True, # Read-only view
            key="dq_table_6"
        )
        st.caption("💡 Use the table headers to sort or filter specific columns. High Completeness ensures better AI modeling.")

# --- PHASE 2: AI MODELING ---
st.write("---")
st.subheader("4️⃣ AI Silver Model Generation")

if not st.session_state.profiles:
    st.info("Please run profiling above to unlock AI modeling.")
else:
    if st.button("🚀 Start AI Silver Modeling", use_container_width=True):
        monitor_placeholder = st.empty()
        progress = st.progress(0)
        
        pipeline_steps = [
            "1. AI Description Generation",
            "2. Semantic Clustering",
            "3. Silver Model Logic Design",
            "4. Snowflake DDL Emission"
        ]

        def update_monitor(active_idx):
            with monitor_placeholder.container(border=True):
                st.write("### 🛤️ AI Pipeline Process Monitor")
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
            all_profiles = st.session_state.profiles
            
            # Convert to "Metadata" format required by description generator
            metadata_struct = {}
            for t, p in all_profiles.items():
                metadata_struct[t] = {"columns": [{"name": k, "datatype": v.get("datatype")} for k, v in p.items()]}
            
            # 1. DESCRIPTIONS
            update_monitor(0)
            progress.progress(0.25)
            desc_data = generate_descriptions_cached(metadata_struct, all_profiles)
            st.session_state.desc_data = desc_data
            
            # 2. CLUSTERING
            update_monitor(1)
            progress.progress(0.5)
            texts = build_texts(desc_data)
            labels = [t.split(':', 1)[0] for t in texts]
            
            try:
                embeds = get_embeddings_cached(texts)
            except Exception as e:
                st.warning(f"Snowflake Embed failed ({e}), falling back to TF-IDF")
                embeds = fallback_embeddings(texts)
                
            clusters = cluster(embeds, labels, k=max(2, int(len(texts)**0.5)))
            cluster_payload = {'embed_model': 'ui-run', 'cluster_count': len(clusters), 'clusters': clusters}
            
            # 3. SILVER MODEL
            update_monitor(2)
            progress.progress(0.75)
            model_json_str = generate_silver_model_cached(cluster_payload)
            model_data = parse_robust(model_json_str)
            if isinstance(model_data, str):
                model_data = parse_robust(model_data)
            st.session_state.model_data = model_data

            if not isinstance(model_data, dict) or "entities" not in model_data:
                st.error("Failed to generate valid model JSON from LLM.")
                st.stop()
                
            # 4. DDL GENERATION
            update_monitor(3)
            progress.progress(0.9)
            
            ddl_statements = []
            csv_rows = []
            target_schema = "SILVER"
            
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
                    
                    src_cols = attr.get("source_columns", [])
                    if not src_cols: src_cols = ["UNKNOWN.UNKNOWN.UNKNOWN"]
                        
                    for src in src_cols:
                        parts = src.split('.')
                        if len(parts) == 3: s_schema, s_table, s_col = parts
                        elif len(parts) == 2: s_schema, s_table, s_col = schema, parts[0], parts[1]
                        else: s_schema, s_table, s_col = schema, "UNKNOWN", parts[0]
                        
                        s_desc = ""
                        if s_table in desc_data and s_col in desc_data[s_table]:
                            s_desc = desc_data[s_table][s_col]
                        
                        s_type = ""
                        if s_table in all_profiles and s_col in all_profiles[s_table]:
                            s_type = all_profiles[s_table][s_col].get("datatype", "")

                        csv_rows.append({
                            "SourceSchema": s_schema, "SourceTableName": s_table, "SourceColumn": s_col,
                            "SourceDescription": s_desc, "SourceDataType": s_type,
                            "TargetSchema": target_schema, "TargetTableName": t_name, "TargetColumn": col,
                            "TargetDataType": dtype, "TargetDescription": desc,
                            "TransformationRule": "Direct Map" if s_col == col else "Renamed/Transformed",
                            "MappingRationale": attr.get("rationale", "Standard business mapping")
                        })
                
                pk_clause = f",\n    PRIMARY KEY ({', '.join(pk_cols)})" if pk_cols else ""
                full_name = f"{database}.{target_schema}.{t_name}"
                ddl = f"CREATE TRANSIENT TABLE IF NOT EXISTS {full_name} (\n" + ",\n".join(attrs_sql) + pk_clause + "\n);"
                ddl_statements.append(ddl)
            
            progress.progress(1.0)
            update_monitor(4)
            st.success("✅ AI Modeling Complete!")
            st.session_state.final_sql = "\n\n".join(ddl_statements)
            st.session_state.csv_rows = csv_rows

        except Exception as e:
            st.error(f"Pipeline Failed: {e}")
            st.exception(e)

# --- RESULTS ---
if "final_sql" in st.session_state:
    st.write("---")
    st.subheader("5️⃣ Generated Artifacts")
    
    tab1, tab2, tab_logic, tab3, tab4 = st.tabs(["📄 DDL Script", "🧠 Logical Model (JSON)", "🔍 AI Logic & Clustering", "📥 Model Export", "🔗 Lineage"])
    
    with tab1:
        st.code(st.session_state.final_sql, language="sql")
        st.download_button("Download DDL", st.session_state.final_sql, "silver_model.sql", "text/plain")
        
    with tab2:
        st.json(st.session_state.model_data)
        
    with tab_logic:
        st.info("The AI identifies 'Clusters' based on semantic similarity of column descriptions and profiles before generating the final Silver model.")
        
        # Load semantic clusters if file exists
        clusters_file = base_dir / "data" / "real_run" / "profile" / "semantic_clusters.json"
        if clusters_file.exists():
            try:
                clusters_data = json.loads(clusters_file.read_text(encoding='utf-8'))
                st.write(f"**Embed Model Used:** `{clusters_data.get('embed_model', 'N/A')}`")
                st.write(f"**Found {clusters_data.get('cluster_count', 0)} Semantic Groups**")
                
                for idx, columns in clusters_data.get('clusters', {}).items():
                    with st.expander(f"Group {idx}: {len(columns)} Related Columns"):
                        for c in columns:
                            st.write(f"- `{c}`")
            except Exception as e:
                st.error(f"Error loading clusters: {e}")
        else:
            st.warning("Semantic clusters file not found. Ensure the pipeline completed successfully.")

    with tab3:
        if st.session_state.csv_rows:
            df_export = pd.DataFrame(st.session_state.csv_rows)
            st.dataframe(df_export, use_container_width=True)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button("Download Mapping CSV", csv, "silver_mapping.csv", "text/csv")

    with tab4:
        unique_links = set((r['SourceTableName'], r['TargetTableName']) for r in st.session_state.csv_rows)
        if unique_links:
            mermaid_lines = ["graph LR", "  subgraph Bronze"]
            for s_table in sorted(set(r[0] for r in unique_links)):
                s_id = s_table.replace(".", "_").replace(" ", "_")
                mermaid_lines.append(f"    B_{s_id}[\"{s_table}\"]")
            mermaid_lines.append("  end\n  subgraph Silver")
            for t_table in sorted(set(r[1] for r in unique_links)):
                t_id = t_table.replace(".", "_").replace(" ", "_")
                mermaid_lines.append(f"    S_{t_id}[\"{t_table}\"]")
            mermaid_lines.append("  end")
            for s_table, t_table in sorted(unique_links):
                s_id = s_table.replace(".", "_").replace(" ", "_")
                t_id = t_table.replace(".", "_").replace(" ", "_")
                mermaid_lines.append(f"  B_{s_id} --> S_{t_id}")
            st.markdown(f"```mermaid\n" + "\n".join(mermaid_lines) + "\n```")
