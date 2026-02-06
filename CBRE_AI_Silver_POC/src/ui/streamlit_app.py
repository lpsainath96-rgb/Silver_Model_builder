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

from extract.profile_real_data import list_tables, list_schemas, get_connection, get_column_info
from ai.bronze_silver_mapper import map_bronze_to_silver
from ai.iterative_mapper import run_iterative_mapping, generate_column_descriptions, compute_embedding_similarity



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

st.set_page_config(page_title="Data Modeling with AI", layout="wide", page_icon="🤖")
st.title("🤖 Data Modeler  Agent")

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

@st.cache_data(ttl=600, show_spinner=False)
def list_schemas_cached(account, user, role, warehouse, database):
    """List all schemas in the database."""
    os.environ["SNOWFLAKE_ACCOUNT"] = account or ""
    os.environ["SNOWFLAKE_USER"] = user or ""
    os.environ["SNOWFLAKE_ROLE"] = role or ""
    os.environ["SNOWFLAKE_WAREHOUSE"] = warehouse or ""
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    conn = get_connection()
    try:
        return list_schemas(conn)
    finally:
        conn.close()

@st.cache_data(ttl=600, show_spinner=False)
def list_tables_in_schema_cached(account, user, role, warehouse, database, schema):
    """List tables in a specific schema."""
    os.environ["SNOWFLAKE_ACCOUNT"] = account or ""
    os.environ["SNOWFLAKE_USER"] = user or ""
    os.environ["SNOWFLAKE_ROLE"] = role or ""
    os.environ["SNOWFLAKE_WAREHOUSE"] = warehouse or ""
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    conn = get_connection()
    try:
        return list_tables(conn, schema=schema, limit=200)
    finally:
        conn.close()

# --- SIDEBAR: CONNECTION ---
st.sidebar.header("🔌 Snowflake Connection")

# Initialize sidebar values from environment ONLY ONCE on first load
if "sidebar_account" not in st.session_state:
    st.session_state.sidebar_account = os.getenv("SNOWFLAKE_ACCOUNT", "")
if "sidebar_user" not in st.session_state:
    st.session_state.sidebar_user = os.getenv("SNOWFLAKE_USER", "")
if "sidebar_role" not in st.session_state:
    st.session_state.sidebar_role = os.getenv("SNOWFLAKE_ROLE", "")
if "sidebar_warehouse" not in st.session_state:
    st.session_state.sidebar_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "")
if "sidebar_database" not in st.session_state:
    st.session_state.sidebar_database = os.getenv("SNOWFLAKE_DATABASE", "")
if "sidebar_schema" not in st.session_state:
    st.session_state.sidebar_schema = os.getenv("SNOWFLAKE_SCHEMA", "BRONZE")

# Use session state as the source of truth (NOT os.environ)
account = st.sidebar.text_input("❄️ Account", value=st.session_state.sidebar_account, key="account_input")
user = st.sidebar.text_input("👤 User", value=st.session_state.sidebar_user, key="user_input")
role = st.sidebar.text_input("🔑 Role", value=st.session_state.sidebar_role, key="role_input")
warehouse = st.sidebar.text_input("⚙️ Warehouse", value=st.session_state.sidebar_warehouse, key="warehouse_input")
database = st.sidebar.text_input("🗄️ Database", value=st.session_state.sidebar_database, key="database_input")
schema = st.sidebar.text_input("📂 Source Schema", value=st.session_state.sidebar_schema, key="schema_input", help="Schema containing Source tables")

# Update session state when user changes values
st.session_state.sidebar_account = account
st.session_state.sidebar_user = user
st.session_state.sidebar_role = role
st.session_state.sidebar_warehouse = warehouse
st.session_state.sidebar_database = database
st.session_state.sidebar_schema = schema

# CRITICAL FIX: Snapshot Bronze database & schema NOW before Silver can pollute os.environ
# Store in session state to ensure they're preserved across reruns
if "bronze_db_snapshot" not in st.session_state:
    st.session_state.bronze_db_snapshot = database
if "bronze_schema_snapshot" not in st.session_state:
    st.session_state.bronze_schema_snapshot = schema

# Update snapshot only if sidebar values actually changed (user edited them)
if database != "":
    st.session_state.bronze_db_snapshot = database
if schema != "":
    st.session_state.bronze_schema_snapshot = schema


if st.sidebar.button("⚡ Test Connection", use_container_width=True):
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

from profiling.generate_long_descriptions import generate_descriptions
from ai.semantic_clustering import build_texts, get_embeddings_snowflake, cluster, fallback_embeddings
from ai.silver_model_generator import generate_silver_model
from ai.generate_silver_snowflake_ddl import parse_robust, sanitize, build_column_line, DEFAULT_PK_CANDIDATES
from extract.profile_real_data import profile_table

# --- SESSION STATE ---
if "profiles" not in st.session_state:
    st.session_state.profiles = None
if "model_data" not in st.session_state:
    st.session_state.model_data = None
if "desc_data" not in st.session_state:
    st.session_state.desc_data = None


# --- MODE TOGGLE ---

st.subheader("🎯 Mode Selection")
mode = st.radio(
    "Choose pipeline mode:",
    ["🔨 Generate New Model", "🔗 Map to Existing Target Schema"],
    horizontal=True,
    key="pipeline_mode"
)

is_mapping_mode = mode == "🔗 Map to Existing Target Schema"

st.subheader("📥 1. Select Source Tables")

# Use the protected snapshot values
bronze_db = st.session_state.bronze_db_snapshot
bronze_schema = st.session_state.bronze_schema_snapshot

# Display current Bronze context
st.caption(f"📍 **Bronze Context:** `{bronze_db}.{bronze_schema}` (from sidebar connection)")

# Force reset environment to Bronze context
os.environ["SNOWFLAKE_DATABASE"] = bronze_db
os.environ["SNOWFLAKE_SCHEMA"] = bronze_schema

# Fetch tables with explicit parameters (not relying on environment)
tables = []
try:
    tables = list_tables_cached(account, user, role, warehouse, bronze_db, bronze_schema)
    st.caption(f"🔍 Debug: Found {len(tables)} tables in `{bronze_db}.{bronze_schema}`")
except Exception as e:
    st.info(f"Please configure connection in sidebar to see tables. Error: {e}")

# Store Bronze tables in session state
if "bronze_selected" not in st.session_state:
    st.session_state.bronze_selected = []

# Create a unique key based on database and schema
bronze_context_key = f"{bronze_db}_{bronze_schema}"
if "bronze_context" not in st.session_state:
    st.session_state.bronze_context = bronze_context_key

# If context changed, reset selections
if st.session_state.bronze_context != bronze_context_key:
    st.session_state.bronze_selected = []
    st.session_state.bronze_context = bronze_context_key

# Filter to only valid selections
valid_bronze = [t for t in st.session_state.bronze_selected if t in tables]

# Debug info
if len(st.session_state.bronze_selected) > 0:
    st.caption(f"🔍 Debug: Session has {len(st.session_state.bronze_selected)} bronze tables, {len(valid_bronze)} are valid")

selected_tables = st.multiselect(
    "Choose Source tables:", 
    tables,
    default=valid_bronze,
    key="bronze_multiselect"
)

# Store selection immediately
st.session_state.bronze_selected = selected_tables


# --- SILVER SCHEMA SELECTION (Mapping Mode Only) ---
selected_silver_schema = None
selected_silver_tables = []

if is_mapping_mode:
    st.subheader("📤 1.b Select Target (Database/Schema/Tables)")
    st.info("💡 **Tip:** Target tables can be in a different database than Source. Specify the target database and schema below.")
    
    col_db, col_schema = st.columns(2)
    
    with col_db:
        # Allow selecting a different database for Silver target
        silver_database = st.text_input(
            "Silver Database:", 
            value=database, 
            help="Database containing Silver target tables (can be different from Bronze)",
            key="silver_db_input"
        )
        # Store in session state for later use
        st.session_state['silver_database'] = silver_database
    
    with col_schema:
        try:
            # Save original database to restore after
            original_db = os.environ.get("SNOWFLAKE_DATABASE", database)
            
            # Temporarily set to silver database for this query
            os.environ["SNOWFLAKE_DATABASE"] = silver_database
            available_schemas = list_schemas_cached(account, user, role, warehouse, silver_database)
            
            # Restore original database immediately
            os.environ["SNOWFLAKE_DATABASE"] = original_db
            
            selected_silver_schema = st.selectbox(
                "Silver Schema:", 
                available_schemas, 
                index=0 if available_schemas else 0,
                key="silver_schema_selection"
            )
        except Exception as e:
            st.warning(f"Could not load Silver schemas from `{silver_database}`: {e}")
            available_schemas = []
            selected_silver_schema = st.text_input("Silver Schema (manual):", value="SILVER", key="silver_schema_manual")
            # Ensure we restore even on error
            os.environ["SNOWFLAKE_DATABASE"] = database
    
    if selected_silver_schema:
        try:
            # Save original database
            original_db = os.environ.get("SNOWFLAKE_DATABASE", database)
            
            # Temporarily set to silver database
            os.environ["SNOWFLAKE_DATABASE"] = silver_database
            silver_tables = list_tables_in_schema_cached(account, user, role, warehouse, silver_database, selected_silver_schema)
            
            # Restore original database immediately
            os.environ["SNOWFLAKE_DATABASE"] = original_db
            
            # Store Silver tables in a more stable way
            if "silver_selected" not in st.session_state:
                st.session_state.silver_selected = []
            
            # Create context key for Silver
            silver_context_key = f"{silver_database}_{selected_silver_schema}"
            if "silver_context" not in st.session_state:
                st.session_state.silver_context = silver_context_key
            
            # If context changed, reset selections
            if st.session_state.silver_context != silver_context_key:
                st.session_state.silver_selected = []
                st.session_state.silver_context = silver_context_key
            
            # Filter to only valid selections
            valid_silver = [t for t in st.session_state.silver_selected if t in silver_tables]
            
            selected_silver_tables = st.multiselect(
                "Choose Silver tables to map to:", 
                silver_tables,
                default=valid_silver,
                key="silver_multiselect"
            )
            
            # Store selection immediately
            st.session_state.silver_selected = selected_silver_tables
            
        except Exception as e:
            st.warning(f"Could not load Silver tables: {e}")
            # Ensure we restore even on error
            os.environ["SNOWFLAKE_DATABASE"] = database

    
    # Store silver database for later use
    st.session_state.silver_database = silver_database




# --- COLUMN SELECTION ---
selected_columns_map = {} # {table: [col1, col2]}


if selected_tables:
    st.subheader("2️⃣ Select Required Columns")
    
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
if is_mapping_mode:
    st.subheader("📊 3. Profile Source Data")
else:
    st.subheader("📊 3. Profiling")

col1, col2 = st.columns([2, 1])
with col2:
    sample_pct = st.slider("Sample Rate (%)", 1, 100, 100, help="For tables with millions of rows, use a lower sample rate (e.g., 10%) for instant results.")

profile_btn_label = "🔍 Profile Bronze Data" if is_mapping_mode else "🔍 Profile Data"
if st.button(profile_btn_label, use_container_width=True):
    if not selected_tables:
        st.warning("Please select at least one table.")
    else:
        status_msg = "🛠️ Analyzing Bronze Source Data..." if is_mapping_mode else "🛠️ Running Data Quality Analysis..."
        with st.status(status_msg, expanded=True) as status:

            conn = None
            all_profiles = {}
            try:
                # Use Bronze snapshot values to ensure correct database context
                bronze_db = st.session_state.bronze_db_snapshot
                bronze_schema = st.session_state.bronze_schema_snapshot
                
                # Set environment to Bronze context before profiling
                os.environ["SNOWFLAKE_DATABASE"] = bronze_db
                os.environ["SNOWFLAKE_SCHEMA"] = bronze_schema
                
                conn = get_connection()
                total_cols_profiled = 0
                for table in selected_tables:
                    target_cols = selected_columns_map.get(table, [])
                    if not target_cols: continue
                    st.write(f"Profiling `{bronze_db}.{bronze_schema}.{table}` ({len(target_cols)} columns, {sample_pct}% sample)...")
                    prof = profile_table_cached(account, user, role, warehouse, bronze_db, bronze_schema, table, tuple(target_cols), sample_pct)
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
    if is_mapping_mode:
        st.subheader("📊 Bronze Source Analysis")
    else:
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

if is_mapping_mode:
    # --- MAPPING MODE PIPELINE ---
    st.subheader("4️⃣ AI Source-to-Target Mapping")
    
    if not st.session_state.profiles:
        st.info("Please run Bronze profiling above first.")
    elif not selected_silver_tables:
        st.info("Please select Silver target tables above.")
    else:
        # Business Context Input
        st.markdown("**📝 Business Context (Optional)**")
        business_context = st.text_area(
            "Describe how the Source tables relate to Target, naming conventions, or any mapping hints:",
            placeholder="Example: The Source tables are different CRM systems. The Target table is a unified customer view. CUST_ID maps to CUSTOMER_KEY, EMAIL maps to EMAIL_ADDRESS...",
            height=100,
            key="business_context"
        )
        
        if st.button("🧠 Run AI Mapping", use_container_width=True):
            with st.status("🔍 Running Iterative AI Mapping...", expanded=True) as status:
                try:
                    # 1. Profile Silver tables - use silver database
                    st.write("📊 Profiling target tables...")
                    silver_profiles = {}
                    silver_db = st.session_state.get('silver_database', database)
                    conn = get_connection()
                    for s_table in selected_silver_tables:
                        os.environ["SNOWFLAKE_DATABASE"] = silver_db
                        os.environ["SNOWFLAKE_SCHEMA"] = selected_silver_schema
                        prof = profile_table_cached(account, user, role, warehouse, silver_db, selected_silver_schema, s_table, (), 100)
                        silver_profiles[s_table] = prof
                        st.write(f"  ✓ Profiled `{silver_db}.{selected_silver_schema}.{s_table}`")
                    conn.close()

                    
                    # 2. Count columns for progress
                    total_target_cols = sum(len(cols) for cols in silver_profiles.values() if isinstance(cols, dict))
                    total_source_cols = sum(len(cols) for cols in st.session_state.profiles.values() if isinstance(cols, dict))
                    st.write(f"🎯 Target: {total_target_cols} Silver columns | Source: {total_source_cols} Bronze columns")
                    
                    # 3. Generate semantic descriptions for better matching
                    st.write("📝 Generating semantic descriptions for columns...")
                    
                    bronze_descriptions = generate_column_descriptions(
                        st.session_state.profiles, 
                        schema
                    )
                    st.write(f"  ✓ Source descriptions: {sum(len(v) for v in bronze_descriptions.values())} columns")
                    
                    silver_descriptions = generate_column_descriptions(
                        silver_profiles, 
                        selected_silver_schema
                    )
                    st.write(f"  ✓ Target descriptions: {sum(len(v) for v in silver_descriptions.values())} columns")
                    
                    # 4. Compute embedding similarity for candidate filtering
                    st.write("🔗 Computing embedding similarity matrix...")
                    similarity_hints = compute_embedding_similarity(
                        bronze_descriptions, 
                        silver_descriptions
                    )
                    if similarity_hints:
                        st.write(f"  ✓ Similarity computed for {len(similarity_hints)} target columns")
                    
                    # 5. Create progress bar
                    progress_bar = st.progress(0, text="Initializing iterative mapping...")
                    
                    def update_progress(current, total, message):
                        pct = current / total if total > 0 else 0
                        progress_bar.progress(pct, text=message)
                    
                    # 6. Run iterative AI mapping with descriptions and similarity hints
                    st.write("🤖 Running iterative batch mapping (5 target × 15 source per call)...")
                    mappings = run_iterative_mapping(
                        source_profiles=st.session_state.profiles,
                        target_profiles=silver_profiles,
                        source_schema_name=schema,  # Source schema from sidebar
                        target_schema_name=selected_silver_schema,
                        business_context=business_context,
                        progress_callback=update_progress,
                        source_descriptions=bronze_descriptions,
                        target_descriptions=silver_descriptions,
                        similarity_hints=similarity_hints
                    )

                    
                    progress_bar.progress(1.0, text="Mapping complete!")
                    st.session_state.mapping_results = mappings
                    status.update(label=f"✅ Iterative Mapping Complete! ({len(mappings)} mappings)", state="complete", expanded=False)
                    
                except Exception as e:
                    import traceback
                    st.error(f"Mapping Error: {e}")
                    st.code(traceback.format_exc())
                    status.update(label="❌ Mapping Failed", state="error", expanded=True)

        
        # Display mapping results with rich visualizations
        if "mapping_results" in st.session_state and st.session_state.mapping_results:
            st.write("---")
            st.subheader("📊 Mapping Results & Visualization")
            
            df_map = pd.DataFrame(st.session_state.mapping_results)
            
            # Create tabs for different views
            tab_table, tab_sankey, tab_json, tab_justify, tab_export = st.tabs([
                "📋 Mapping Table", 
                "🔀 Sankey Diagram", 
                "🧠 JSON Model",
                "🔍 AI Justifications",
                "📥 Export"
            ])
            
            with tab_table:
                st.write("**Complete Column-Level Mapping**")
                st.dataframe(df_map, use_container_width=True, height=400)
                
                # Statistics
                total_mappings = len(df_map)
                mapped_count = len(df_map[df_map['SourceColumn'] != 'UNMAPPED'])
                unmapped_count = len(df_map[df_map['SourceColumn'] == 'UNMAPPED'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Mappings", total_mappings)
                with col2:
                    st.metric("Mapped Columns", mapped_count, delta=f"{mapped_count/total_mappings*100:.1f}%" if total_mappings else "0%")
                with col3:
                    st.metric("Unmapped Columns", unmapped_count, delta_color="inverse")
            
            with tab_sankey:
                st.write("**Visual Column-Level Mapping Flow**")
                try:
                    import plotly.graph_objects as go
                    
                    # Build Sankey data - only for mapped columns
                    mapped_df = df_map[df_map['SourceColumn'] != 'UNMAPPED'].copy()
                    
                    if len(mapped_df) > 0:
                        # Get unique source and target nodes
                        source_nodes = mapped_df.apply(lambda r: f"{r.get('SourceTable', 'SRC')}.{r.get('SourceColumn', 'col')}", axis=1).unique().tolist()
                        target_nodes = mapped_df.apply(lambda r: f"{r.get('TargetTable', 'TGT')}.{r.get('TargetColumn', 'col')}", axis=1).unique().tolist()
                        
                        # Create node list: sources first, then targets
                        all_nodes = source_nodes + target_nodes
                        node_indices = {node: i for i, node in enumerate(all_nodes)}
                        
                        # Build links
                        sources = []
                        targets = []
                        values = []
                        hover_texts = []
                        
                        for _, row in mapped_df.iterrows():
                            src_node = f"{row.get('SourceTable', 'SRC')}.{row.get('SourceColumn', 'col')}"
                            tgt_node = f"{row.get('TargetTable', 'TGT')}.{row.get('TargetColumn', 'col')}"
                            
                            if src_node in node_indices and tgt_node in node_indices:
                                sources.append(node_indices[src_node])
                                targets.append(node_indices[tgt_node])
                                values.append(1)
                                hover_texts.append(row.get('Justification', '')[:100])
                        
                        # Colors: blue for source, green for target
                        node_colors = ['#3498db'] * len(source_nodes) + ['#27ae60'] * len(target_nodes)
                        
                        fig = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=all_nodes,
                                color=node_colors
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                customdata=hover_texts,
                                hovertemplate='%{customdata}<extra></extra>'
                            )
                        )])
                        
                        fig.update_layout(
                            title_text="Bronze → Silver Column Mapping Flow",
                            font_size=10,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No mapped columns to display in Sankey diagram.")
                        
                except ImportError:
                    st.warning("Install plotly for Sankey diagram: `pip install plotly`")
                    st.info("Falling back to Mermaid diagram...")
                    
                    # Mermaid fallback
                    unique_links = set()
                    for _, row in df_map.iterrows():
                        if row.get('SourceColumn') != 'UNMAPPED':
                            unique_links.add((row.get('SourceTable', 'SRC'), row.get('TargetTable', 'TGT')))
                    
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
            
            with tab_json:
                st.write("**Raw Mapping Model (JSON Format)**")
                
                # Convert flat results to nested JSON structure
                nested_model = {"mapped_columns": []}
                
                # Group by target column
                for target_col in df_map['TargetColumn'].unique():
                    target_rows = df_map[df_map['TargetColumn'] == target_col]
                    first_row = target_rows.iloc[0]
                    
                    source_mappings = []
                    for _, row in target_rows.iterrows():
                        if row.get('SourceColumn') != 'UNMAPPED':
                            source_mappings.append({
                                "source_column_name": row.get('SourceColumn', ''),
                                "source_table_name": row.get('SourceTable', ''),
                                "source_schema_name": row.get('SourceSchema', ''),
                                "justification": row.get('Justification', '')
                            })
                    
                    nested_model["mapped_columns"].append({
                        "target_column_name": target_col,
                        "target_table_name": first_row.get('TargetTable', ''),
                        "target_schema_name": first_row.get('TargetSchema', ''),
                        "source_column_name": source_mappings,
                        "final_transformation_logic": first_row.get('TransformationLogic', ''),
                        "work_notes": first_row.get('WorkNotes', '')
                    })
                
                st.json(nested_model)
                
                # Download JSON
                json_str = json.dumps(nested_model, indent=2)
                st.download_button(
                    "📥 Download JSON Model",
                    json_str,
                    "mapping_model.json",
                    "application/json"
                )
            
            with tab_justify:
                st.write("**AI Justifications & Transformation Logic**")
                
                # Group by target table for easier navigation
                for target_table in df_map['TargetTable'].unique():
                    table_rows = df_map[df_map['TargetTable'] == target_table]
                    
                    with st.expander(f"🎯 **{target_table}** ({len(table_rows)} columns)", expanded=True):
                        for _, row in table_rows.iterrows():
                            source_info = f"{row.get('SourceTable', 'N/A')}.{row.get('SourceColumn', 'N/A')}"
                            target_info = row.get('TargetColumn', 'N/A')
                            
                            if row.get('SourceColumn') == 'UNMAPPED':
                                st.markdown(f"❌ **{target_info}** → `UNMAPPED`")
                            else:
                                st.markdown(f"✅ **{target_info}** ← `{source_info}`")
                                
                                if row.get('Justification'):
                                    st.caption(f"💡 *{row.get('Justification')}*")
                                
                                if row.get('TransformationLogic'):
                                    st.code(row.get('TransformationLogic'), language="sql")
                                
                                if row.get('WorkNotes') and row.get('WorkNotes') != 'Awaiting initial scan.':
                                    st.info(f"📝 Work Notes: {row.get('WorkNotes')}")
                            
                            st.markdown("---")
            
            with tab_export:
                st.write("**Export Options**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = df_map.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full CSV",
                        data=csv_data,
                        file_name="bronze_to_silver_mapping.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Mapped only CSV
                    mapped_only = df_map[df_map['SourceColumn'] != 'UNMAPPED']
                    csv_mapped = mapped_only.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Mapped Only CSV",
                        data=csv_mapped,
                        file_name="bronze_to_silver_mapped_only.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.write("---")
                st.write("**Quick Stats:**")
                st.write(f"- Total target columns: **{len(df_map['TargetColumn'].unique())}**")
                st.write(f"- Source tables used: **{len(df_map[df_map['SourceTable'] != 'UNMAPPED']['SourceTable'].unique())}**")
                st.write(f"- Target tables: **{len(df_map['TargetTable'].unique())}**")


else:
    # --- ORIGINAL GENERATION MODE PIPELINE ---
    st.subheader("🪄 4. AI Model Generation")
    
    if not st.session_state.profiles:
        st.info("Please run profiling above to unlock AI modeling.")
    else:
        if st.button("🪄 Start AI Model", use_container_width=True):
            monitor_placeholder = st.empty()
            progress = st.progress(0)
            
            pipeline_steps = [
                "1. AI Description Generation",
                "2. Semantic Clustering",
                "3. Model Logic Design",
                "4. Snowflake DDL Generation"
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
