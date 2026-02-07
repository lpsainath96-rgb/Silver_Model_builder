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

from extract.profile_real_data import list_tables, get_connection, get_column_info, profile_table, classify_column_type

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

@st.cache_data(ttl=600, show_spinner=False)
def list_tables_cached(account, user, role, warehouse, database, schema):
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
    os.environ["SNOWFLAKE_DATABASE"] = database or ""
    os.environ["SNOWFLAKE_SCHEMA"] = schema or ""
    conn = get_connection()
    try:
        return get_column_info(conn, table_name)
    finally:
        conn.close()

@st.cache_data(ttl=3600, show_spinner=False)
def profile_table_cached(account, user, role, warehouse, database, schema, table_name, target_columns, sample_pct=100):
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
    import uuid, shutil
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
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@st.cache_data(show_spinner=False)
def get_embeddings_cached(texts):
    return get_embeddings_snowflake(texts)

@st.cache_data(show_spinner=False)
def generate_silver_model_cached(cluster_payload):
    return generate_silver_model(cluster_payload)

@st.cache_data(show_spinner=False)
def deep_profile_cached(table, profiles_json, descriptions_json, sample_data_json, business_contexts_json):
    """Cached wrapper for deep column profiling."""
    conn = get_connection()
    try:
        profiles = json.loads(profiles_json)
        descriptions = json.loads(descriptions_json)
        sample_data = json.loads(sample_data_json)
        business_contexts = json.loads(business_contexts_json)
        return deep_profile_table(conn, table, profiles, descriptions, sample_data, business_contexts)
    finally:
        conn.close()

# --- IMPORTS ---
from profiling.generate_long_descriptions import generate_descriptions
from ai.semantic_clustering import build_texts, get_embeddings_snowflake, cluster, fallback_embeddings
from ai.silver_model_generator import generate_silver_model
from ai.generate_silver_snowflake_ddl import parse_robust, sanitize, build_column_line, DEFAULT_PK_CANDIDATES
from ai.deep_column_profiler import deep_profile_table, execute_prescribed_checks
from profiling.visualization_engine import (
    build_completeness_chart, build_datatype_distribution_chart,
    build_null_heatmap, build_string_length_histogram,
    build_value_frequency_chart, build_quality_score_gauge,
)
from profiling.report_generator import generate_html_report

# --- SIDEBAR: CONNECTION ---
st.sidebar.header("Snowflake Connection")

account = st.sidebar.text_input("Account", value=os.getenv("SNOWFLAKE_ACCOUNT", ""))
user = st.sidebar.text_input("User", value=os.getenv("SNOWFLAKE_USER", ""))
password = st.sidebar.text_input("Password", value=os.getenv("SNOWFLAKE_PASSWORD", ""), type="password")
role = st.sidebar.text_input("Role", value=os.getenv("SNOWFLAKE_ROLE", ""))
warehouse = st.sidebar.text_input("Warehouse", value=os.getenv("SNOWFLAKE_WAREHOUSE", ""))
database = st.sidebar.text_input("Database", value=os.getenv("SNOWFLAKE_DATABASE", ""))
schema = st.sidebar.text_input("Schema", value=os.getenv("SNOWFLAKE_SCHEMA", "SILVER"))

if st.sidebar.button("Test Connection"):
    try:
        os.environ["SNOWFLAKE_ACCOUNT"] = account
        os.environ["SNOWFLAKE_USER"] = user
        os.environ["SNOWFLAKE_PASSWORD"] = password or os.getenv("SNOWFLAKE_PASSWORD", "")
        os.environ["SNOWFLAKE_ROLE"] = role
        os.environ["SNOWFLAKE_WAREHOUSE"] = warehouse
        os.environ["SNOWFLAKE_DATABASE"] = database
        os.environ["SNOWFLAKE_SCHEMA"] = schema
        conn = get_connection()
        st.sidebar.success("Connected!")
        conn.close()
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")

# --- SESSION STATE ---
for key in ["profiles", "model_data", "desc_data", "sample_data",
            "deep_profiles", "business_contexts", "prescribed_check_results",
            "report_html", "final_sql", "csv_rows", "selected_tables", "selected_columns_map"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.business_contexts is None:
    st.session_state.business_contexts = {}

# --- MAIN TABS ---
tab_data, tab_profiling, tab_deep, tab_silver, tab_reports = st.tabs([
    "Data Selection",
    "Profiling & Quality",
    "AI Deep Analysis",
    "Silver Model",
    "Reports & Export",
])

# =============================================
# TAB 1: DATA SELECTION
# =============================================
with tab_data:
    st.subheader("1. Select Bronze Tables")

    tables = []
    try:
        tables = list_tables_cached(account, user, role, warehouse, database, schema)
    except Exception:
        st.info("Please configure connection in sidebar to see tables.")

    selected_tables = st.multiselect("Choose tables:", tables)
    st.session_state.selected_tables = selected_tables

    # Column selection
    selected_columns_map = {}

    if selected_tables:
        st.subheader("2. Select Target Columns")

        for table in selected_tables:
            with st.expander(f"Settings for `{table}`", expanded=False):
                cols_info = get_column_info_cached(account, user, role, warehouse, database, schema, table)
                col_names = [c['name'] for c in cols_info]

                df_cols = pd.DataFrame({
                    "Select": [True] * len(col_names),
                    "Column Name": col_names
                })

                edited_df = st.data_editor(
                    df_cols,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Include", help="Check to include in profiling", default=True),
                        "Column Name": st.column_config.TextColumn("Column", width="medium", disabled=True)
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=f"editor_{table}"
                )

                picked = edited_df[edited_df["Select"]]["Column Name"].tolist()
                selected_columns_map[table] = picked

    st.session_state.selected_columns_map = selected_columns_map

    st.write("---")

    # Profiling trigger
    col1, col2 = st.columns([2, 1])
    with col2:
        sample_pct = st.slider("Sample Rate (%)", 1, 100, 100, help="For large tables, use a lower sample rate for faster results.")

    if st.button("Profile Data", use_container_width=True):
        if not selected_tables:
            st.warning("Please select at least one table.")
        else:
            with st.status("Running Data Quality Analysis...", expanded=True) as status:
                conn = None
                all_profiles = {}
                try:
                    conn = get_connection()
                    total_cols_profiled = 0
                    for table in selected_tables:
                        target_cols = selected_columns_map.get(table, [])
                        if not target_cols:
                            continue
                        st.write(f"Profiling `{table}` ({len(target_cols)} columns, {sample_pct}% sample)...")
                        prof = profile_table_cached(account, user, role, warehouse, database, schema, table, tuple(target_cols), sample_pct)
                        all_profiles[table] = prof
                        total_cols_profiled += len(prof)

                    st.session_state.profiles = all_profiles

                    # Extract sample data into session state
                    sample_data = {}
                    for tbl, cols in all_profiles.items():
                        for col_name, stats in cols.items():
                            sample_data[col_name] = stats.get("sample_values", [])
                    st.session_state.sample_data = sample_data

                    if total_cols_profiled > 0:
                        status.update(label=f"Profiling Complete! ({total_cols_profiled} columns analyzed)", state="complete", expanded=False)
                    else:
                        status.update(label="Profiling finished, but no columns were analyzed.", state="complete", expanded=True)
                        st.warning("No columns were analyzed. Ensure you checked the 'Select' box for columns in step 2.")
                except Exception as e:
                    st.error(f"Profiling Error: {str(e)}")
                    status.update(label="Profiling Failed", state="error", expanded=True)
                finally:
                    if conn:
                        conn.close()

# =============================================
# TAB 2: PROFILING & QUALITY
# =============================================
with tab_profiling:
    if not st.session_state.profiles:
        st.info("Please run profiling in the 'Data Selection' tab first.")
    else:
        st.subheader("Data Quality Grid")

        # Build DQ dataframe
        all_col_stats = []
        for table, cols in st.session_state.profiles.items():
            for col_name, stats in cols.items():
                total = stats.get('total', 0)
                nulls = stats.get('nulls', 0)
                completeness = (total - nulls) / total * 100 if total > 0 else 0
                rng = stats.get('range', {})
                min_val = rng.get('min', 'N/A')
                max_val = rng.get('max', 'N/A')
                top_vals = stats.get('top_values', [])
                if top_vals and isinstance(top_vals[0], dict):
                    top_str = ", ".join([f"{v['value']} ({v['count']})" for v in top_vals[:3]])
                else:
                    top_str = ", ".join([str(v) for v in top_vals[:3]])

                row_data = {
                    "Table": table,
                    "Column": col_name,
                    "Type": stats.get('datatype', 'UNKNOWN'),
                    "Category": stats.get('type_category', ''),
                    "Completeness": completeness,
                    "Null %": (nulls / total * 100) if total > 0 else 0,
                    "Distinct": stats.get('distinct', 0),
                    "Min": min_val,
                    "Max": max_val,
                    "Top Values": top_str,
                }

                # String stats columns
                ss = stats.get('string_stats')
                if ss:
                    row_data["Min Len"] = ss.get('min_length', '')
                    row_data["Max Len"] = ss.get('max_length', '')
                    row_data["Avg Len"] = ss.get('avg_length', '')
                    upper_pct = ss.get('upper_case_pct', 0)
                    lower_pct = ss.get('lower_case_pct', 0)
                    if upper_pct > 80:
                        row_data["Case"] = "UPPER"
                    elif lower_pct > 80:
                        row_data["Case"] = "lower"
                    else:
                        row_data["Case"] = "Mixed"
                else:
                    row_data["Min Len"] = ""
                    row_data["Max Len"] = ""
                    row_data["Avg Len"] = ""
                    row_data["Case"] = ""

                all_col_stats.append(row_data)

        df_dq = pd.DataFrame(all_col_stats)

        if df_dq.empty:
            st.info("No profiling data available.")
        else:
            # Summary Metrics Header
            m1, m2, m3, m4, m5 = st.columns(5)
            avg_comp = df_dq["Completeness"].mean()
            total_records = sum(stats.get('total', 0) for table_stats in st.session_state.profiles.values() for stats in table_stats.values())
            unique_cells = df_dq["Distinct"].sum()
            grade = "A" if avg_comp > 90 else "B" if avg_comp > 75 else "C" if avg_comp > 50 else "D"

            m1.metric("Overall Health", f"{grade}", help=f"Average Score: {avg_comp:.1f}%")
            m2.metric("Avg Completeness", f"{avg_comp:.1f}%")
            m3.metric("Total Attributes", len(df_dq))
            m4.metric("High Quality (>95%)", len(df_dq[df_dq["Completeness"] > 95]))
            m5.metric("Action Needed (<50%)", len(df_dq[df_dq["Completeness"] < 50]))

            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.caption(f"Total Rows Analyzed: **{total_records:,}**")
            c2.caption(f"Total Unique Values: **{unique_cells:,}**")
            type_counts = df_dq["Type"].value_counts().to_dict()
            type_str = ", ".join([f"**{k}**: {v}" for k, v in type_counts.items()])
            c3.caption(f"Type Distribution: {type_str}")

            # DQ Grid
            st.data_editor(
                df_dq,
                column_config={
                    "Completeness": st.column_config.ProgressColumn("Completeness", format="%.1f%%", min_value=0, max_value=100),
                    "Null %": st.column_config.NumberColumn("Null %", format="%.1f%%"),
                    "Top Values": st.column_config.TextColumn("Top Values", width="large"),
                },
                hide_index=True,
                use_container_width=True,
                disabled=True,
                key="dq_table_main"
            )

        # --- VISUALIZATIONS ---
        st.write("---")
        st.subheader("Visualizations")

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            try:
                chart = build_completeness_chart(st.session_state.profiles)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Completeness chart error: {e}")

        with viz_col2:
            try:
                chart = build_datatype_distribution_chart(st.session_state.profiles)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Type distribution chart error: {e}")

        try:
            chart = build_null_heatmap(st.session_state.profiles)
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Null heatmap error: {e}")

        # --- PER-COLUMN DRILL-DOWN ---
        st.write("---")
        st.subheader("Per-Column Drill-Down")

        all_columns = []
        for table, cols in st.session_state.profiles.items():
            for col_name in cols:
                all_columns.append(f"{table}.{col_name}")

        if all_columns:
            selected_col = st.selectbox("Select a column to inspect:", all_columns)
            if selected_col:
                tbl, cname = selected_col.split(".", 1)
                col_stats = st.session_state.profiles[tbl][cname]

                dc1, dc2, dc3, dc4 = st.columns(4)
                total = col_stats.get('total', 0)
                nulls = col_stats.get('nulls', 0)
                dc1.metric("Total Rows", f"{total:,}")
                dc2.metric("Nulls", f"{nulls:,}")
                dc3.metric("Distinct", f"{col_stats.get('distinct', 0):,}")
                dc4.metric("Completeness", f"{((total-nulls)/total*100):.1f}%" if total else "N/A")

                # String stats detail
                ss = col_stats.get('string_stats')
                if ss:
                    st.write("**String Analysis:**")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Min Length", ss.get('min_length', 'N/A'))
                    sc2.metric("Max Length", ss.get('max_length', 'N/A'))
                    sc3.metric("Avg Length", ss.get('avg_length', 'N/A'))
                    sc4.metric("Empty Strings", ss.get('empty_string_count', 0))

                    st.write(f"Case: **{ss.get('upper_case_pct', 0)}%** UPPER, **{ss.get('lower_case_pct', 0)}%** lower")

                # Detected patterns
                patterns = col_stats.get('detected_patterns', [])
                if patterns:
                    st.write("**Detected Patterns:**")
                    for p in patterns:
                        st.write(f"- {p['pattern']}: {p['match_pct']}%")

                # Sample values
                samples = col_stats.get('sample_values', [])
                if samples:
                    st.write(f"**Sample Values ({len(samples)}):**")
                    st.code(", ".join(samples[:20]))

                # Charts
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    top_vals = col_stats.get('top_values', [])
                    if top_vals:
                        try:
                            chart = build_value_frequency_chart(top_vals, cname)
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            pass

                with chart_col2:
                    if samples and col_stats.get('type_category') == 'string':
                        try:
                            chart = build_string_length_histogram(samples, cname)
                            st.altair_chart(chart, use_container_width=True)
                        except Exception:
                            pass

# =============================================
# TAB 3: AI DEEP ANALYSIS
# =============================================
with tab_deep:
    if not st.session_state.profiles:
        st.info("Please run profiling in the 'Data Selection' tab first.")
    else:
        st.subheader("AI Deep Column Analysis")
        st.write("Optional: Provide business context for columns to improve AI analysis.")

        # Business context inputs
        with st.expander("Business Context (optional)", expanded=False):
            for table, cols in st.session_state.profiles.items():
                st.write(f"**{table}**")
                for col_name in cols:
                    key = f"biz_ctx_{table}_{col_name}"
                    current = st.session_state.business_contexts.get(col_name, "")
                    val = st.text_input(f"{col_name}", value=current, key=key, placeholder="e.g., 'Revenue metric in USD, reported monthly'")
                    if val:
                        st.session_state.business_contexts[col_name] = val

        if st.button("Run AI Deep Analysis", use_container_width=True):
            with st.status("Running AI Deep Analysis...", expanded=True) as status:
                try:
                    conn = get_connection()
                    all_deep = {}

                    for table, cols in st.session_state.profiles.items():
                        st.write(f"Analyzing `{table}` ({len(cols)} columns)...")

                        # Get descriptions for this table
                        desc_data = st.session_state.desc_data or {}
                        table_descs = {}
                        if isinstance(desc_data, dict):
                            tables_section = desc_data.get("tables", desc_data)
                            table_descs = tables_section.get(table, {})

                        sample_data = {}
                        for col_name, stats in cols.items():
                            sample_data[col_name.upper()] = stats.get("sample_values", [])

                        table_deep = deep_profile_table(
                            conn, table, cols, table_descs, sample_data,
                            st.session_state.business_contexts
                        )
                        all_deep.update(table_deep)

                    st.session_state.deep_profiles = all_deep
                    conn.close()

                    success_count = sum(1 for v in all_deep.values() if v.get("_status") == "success")
                    status.update(label=f"AI Deep Analysis Complete! ({success_count}/{len(all_deep)} columns)", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"Deep Analysis Error: {e}")
                    status.update(label="Deep Analysis Failed", state="error", expanded=True)

        # Display results
        if st.session_state.deep_profiles:
            st.write("---")

            # Quality score overview
            try:
                chart = build_quality_score_gauge(st.session_state.deep_profiles)
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                pass

            # Per-column expandable cards
            st.subheader("Column Analysis Details")

            for col_name, dp in st.session_state.deep_profiles.items():
                score = dp.get("data_quality_score")
                score_label = f"{score}/100" if score is not None else "N/A"
                status_icon = "" if dp.get("_status") == "success" else " [error]"

                with st.expander(f"{col_name} - Quality: {score_label}{status_icon}"):
                    if dp.get("_status") == "error":
                        st.error(f"Analysis failed: {dp.get('_error', 'Unknown error')}")
                        continue

                    # Quality issues
                    issues = dp.get("quality_issues", [])
                    if issues:
                        st.write("**Quality Issues:**")
                        for issue in issues:
                            sev = issue.get("severity", "low")
                            color = {"high": "red", "medium": "orange", "low": "green"}.get(sev, "gray")
                            st.markdown(f":{color}[**{sev.upper()}**] {issue.get('issue', '')} - {issue.get('description', '')}")

                    # Format analysis
                    fmt = dp.get("format_analysis", {})
                    if fmt:
                        fc1, fc2, fc3 = st.columns(3)
                        fc1.metric("Dominant Pattern", fmt.get("dominant_pattern", "N/A"))
                        fc2.metric("Case Convention", fmt.get("case_convention", "N/A"))
                        detected = fmt.get("detected_formats", [])
                        fc3.metric("Detected Formats", ", ".join(detected) if detected else "None")

                    # Statistical insights
                    si = dp.get("statistical_insights", {})
                    if si:
                        si1, si2, si3 = st.columns(3)
                        si1.metric("Distribution", si.get("distribution_type", "N/A"))
                        si2.metric("Cardinality", si.get("cardinality_assessment", "N/A"))
                        st.write(f"**Outlier Indicators:** {si.get('outlier_indicators', 'N/A')}")

                    # Silver recommendations
                    recs = dp.get("silver_recommendations", {})
                    if recs:
                        st.write("**Silver Layer Recommendations:**")
                        rc1, rc2, rc3 = st.columns(3)
                        rc1.write(f"Type: `{recs.get('recommended_datatype', 'N/A')}`")
                        rc2.write(f"Name: `{recs.get('recommended_name', 'N/A')}`")
                        rc3.write(f"PK Candidate: {'Yes' if recs.get('is_pk_candidate') else 'No'}")
                        st.code(recs.get("transformation_logic", "Direct copy"), language="sql")

                    # Business interpretation
                    biz = dp.get("business_interpretation", "")
                    if biz:
                        st.info(biz)

            # Prescribed checks
            st.write("---")
            st.subheader("AI-Prescribed Validation Checks")

            all_prescriptions = []
            for col_name, dp in st.session_state.deep_profiles.items():
                for rx in dp.get("additional_profiling_prescriptions", []):
                    all_prescriptions.append({**rx, "_column": col_name})

            if all_prescriptions:
                st.write(f"The AI has prescribed **{len(all_prescriptions)}** additional checks.")

                if st.button("Execute AI-Prescribed Checks", use_container_width=True):
                    with st.status("Executing prescribed checks...", expanded=True) as status:
                        try:
                            conn = get_connection()
                            schema_name = (os.getenv("SNOWFLAKE_SCHEMA") or "PUBLIC").upper()
                            results = []

                            for table in (st.session_state.selected_tables or []):
                                table_rxs = [rx for rx in all_prescriptions]
                                res = execute_prescribed_checks(conn, table, table_rxs, schema_name)
                                results.extend(res)

                            conn.close()
                            st.session_state.prescribed_check_results = results

                            success = sum(1 for r in results if r.get("status") == "success")
                            status.update(label=f"Checks Complete! ({success}/{len(results)} succeeded)", state="complete", expanded=False)
                        except Exception as e:
                            st.error(f"Check execution error: {e}")
                            status.update(label="Check Execution Failed", state="error", expanded=True)

            # Display check results
            if st.session_state.prescribed_check_results:
                for result in st.session_state.prescribed_check_results:
                    check_status = result.get("status", "unknown")
                    name = result.get("check_name", "unnamed")
                    icon = {"success": "white_check_mark", "blocked": "no_entry", "error": "x"}.get(check_status, "question")

                    with st.expander(f":{icon}: {name} ({check_status})"):
                        st.write(f"**Rationale:** {result.get('rationale', 'N/A')}")
                        st.code(result.get("sql", ""), language="sql")

                        if check_status == "success":
                            data = result.get("data", [])
                            if data:
                                st.dataframe(pd.DataFrame(data), use_container_width=True)
                            st.caption(f"Rows returned: {result.get('row_count', 0)}")
                        elif check_status == "blocked":
                            st.warning(f"Blocked: {result.get('reason', 'Security policy')}")
                        elif check_status == "error":
                            st.error(f"Error: {result.get('error', 'Unknown')}")

# =============================================
# TAB 4: SILVER MODEL
# =============================================
with tab_silver:
    if not st.session_state.profiles:
        st.info("Please run profiling first.")
    else:
        st.subheader("AI Silver Model Generation")

        if st.button("Start AI Silver Modeling", use_container_width=True):
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
                    st.write("### AI Pipeline Process Monitor")
                    cols = st.columns(len(pipeline_steps))
                    for i, name in enumerate(pipeline_steps):
                        with cols[i]:
                            if i < active_idx:
                                st.markdown(f"**{name}**\n\nDone")
                            elif i == active_idx:
                                st.markdown(f"**{name}**\n\nRunning...")
                            else:
                                st.markdown(f"**{name}**\n\nPending")
                    if active_idx < len(pipeline_steps):
                        st.write(f"Current Activity: **{pipeline_steps[active_idx]}**")
                    else:
                        st.write("Current Activity: **Pipeline Complete**")

            try:
                all_profiles = st.session_state.profiles

                # Convert to metadata format
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
                        if not src_cols:
                            src_cols = ["UNKNOWN.UNKNOWN.UNKNOWN"]

                        for src in src_cols:
                            parts = src.split('.')
                            if len(parts) == 3:
                                s_schema, s_table, s_col = parts
                            elif len(parts) == 2:
                                s_schema, s_table, s_col = schema, parts[0], parts[1]
                            else:
                                s_schema, s_table, s_col = schema, "UNKNOWN", parts[0]

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
                st.success("AI Modeling Complete!")
                st.session_state.final_sql = "\n\n".join(ddl_statements)
                st.session_state.csv_rows = csv_rows

            except Exception as e:
                st.error(f"Pipeline Failed: {e}")
                st.exception(e)

    # Results display
    if st.session_state.get("final_sql"):
        st.write("---")
        st.subheader("Generated Artifacts")

        art_tab1, art_tab2, art_tab_logic, art_tab3, art_tab4 = st.tabs(["DDL Script", "Logical Model (JSON)", "AI Logic & Clustering", "Model Export", "Lineage"])

        with art_tab1:
            st.code(st.session_state.final_sql, language="sql")
            st.download_button("Download DDL", st.session_state.final_sql, "silver_model.sql", "text/plain")

        with art_tab2:
            st.json(st.session_state.model_data)

        with art_tab_logic:
            st.info("The AI identifies 'Clusters' based on semantic similarity of column descriptions and profiles before generating the final Silver model.")
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

        with art_tab3:
            if st.session_state.csv_rows:
                df_export = pd.DataFrame(st.session_state.csv_rows)
                st.dataframe(df_export, use_container_width=True)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button("Download Mapping CSV", csv, "silver_mapping.csv", "text/csv")

        with art_tab4:
            if st.session_state.csv_rows:
                unique_links = set((r['SourceTableName'], r['TargetTableName']) for r in st.session_state.csv_rows)
                if unique_links:
                    mermaid_lines = ["graph LR", "  subgraph Bronze"]
                    for s_table in sorted(set(r[0] for r in unique_links)):
                        s_id = s_table.replace(".", "_").replace(" ", "_")
                        mermaid_lines.append(f'    B_{s_id}["{s_table}"]')
                    mermaid_lines.append("  end\n  subgraph Silver")
                    for t_table in sorted(set(r[1] for r in unique_links)):
                        t_id = t_table.replace(".", "_").replace(" ", "_")
                        mermaid_lines.append(f'    S_{t_id}["{t_table}"]')
                    mermaid_lines.append("  end")
                    for s_table, t_table in sorted(unique_links):
                        s_id = s_table.replace(".", "_").replace(" ", "_")
                        t_id = t_table.replace(".", "_").replace(" ", "_")
                        mermaid_lines.append(f"  B_{s_id} --> S_{t_id}")
                    st.markdown(f"```mermaid\n" + "\n".join(mermaid_lines) + "\n```")

# =============================================
# TAB 5: REPORTS & EXPORT
# =============================================
with tab_reports:
    if not st.session_state.profiles:
        st.info("Please run profiling first to generate reports.")
    else:
        st.subheader("Reports & Export")

        if st.button("Generate HTML Report", use_container_width=True):
            with st.spinner("Generating report..."):
                # Flatten descriptions
                desc_data = st.session_state.desc_data or {}
                flat_descs = {}
                if isinstance(desc_data, dict):
                    tables_section = desc_data.get("tables", desc_data)
                    for tbl, cols in tables_section.items():
                        if isinstance(cols, dict):
                            flat_descs.update(cols)

                html = generate_html_report(
                    profiles=st.session_state.profiles,
                    descriptions=flat_descs,
                    deep_profiles=st.session_state.deep_profiles or {},
                    metadata={"generated_by": "Silver Model Builder UI"},
                )
                st.session_state.report_html = html
                st.success("Report generated!")

        if st.session_state.report_html:
            # Preview
            st.write("**Report Preview:**")
            st.components.v1.html(st.session_state.report_html, height=600, scrolling=True)

            # Download buttons
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download HTML Report",
                    st.session_state.report_html,
                    "profiling_report.html",
                    "text/html",
                )
            with dl2:
                # Full JSON export
                export_data = {
                    "profiles": st.session_state.profiles,
                    "descriptions": st.session_state.desc_data,
                    "deep_profiles": st.session_state.deep_profiles,
                    "prescribed_check_results": st.session_state.prescribed_check_results,
                }
                export_json = json.dumps(export_data, indent=2, cls=SnowflakeEncoder)
                st.download_button(
                    "Download Full JSON Analysis",
                    export_json,
                    "full_analysis.json",
                    "application/json",
                )
