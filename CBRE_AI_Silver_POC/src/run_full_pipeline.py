"""Run End-to-End Silver Layer Generation Pipeline (Real Data).

This script orchestrates the entire pipeline from real data profiling to Snowflake DDL generation:
1. Profile Real Data (Snowflake -> JSON)
2. Generate AI Descriptions (Using AI_COMPLETE)
3. Semantic Clustering (Using AI_EMBED)
4. Silver Model Generation (Using AI_COMPLETE)
5. Generate Snowflake DDL (JSON -> SQL)
"""
import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def run_step(step_name, command):
    print(f"\n{'='*60}")
    print(f"🚀 Running {step_name}...")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"✅ {step_name} Complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} Failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    base_dir = Path(__file__).resolve().parent.parent
    src_dir = base_dir / "src"
    data_dir = base_dir / "data" / "real_run"
    profile_dir = data_dir / "profile"
    
    # Ensure output directories exist
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    python_exe = sys.executable
    
    # 1. Profile Real Data
    # Outputs: profile.json AND real_data_metadata.json
    profile_out = profile_dir / "column_profiles_real.json"
    metadata_out = profile_dir / "real_data_metadata.json"
    
    run_step("1. Profile Real Data", [
        python_exe, str(src_dir / "extract" / "profile_real_data.py"),
        "-o", str(profile_out)
    ])
    
    # 2. Generate Long Descriptions
    # Inputs: metadata.json, profile.json
    desc_out = profile_dir / "column_descriptions_llm.json"
    
    run_step("2. Generate AI Descriptions", [
        python_exe, str(src_dir / "profiling" / "generate_long_descriptions.py"),
        "--mode", "llm",
        "-m", str(metadata_out),
        "-p", str(profile_out),
        "-o", str(desc_out)
    ])
    
    # 3. Semantic Clustering
    # Input: descriptions.json
    clusters_out = profile_dir / "semantic_clusters.json"
    
    run_step("3. Semantic Clustering", [
        python_exe, str(src_dir / "ai" / "semantic_clustering.py"),
        "-d", str(desc_out),
        "-o", str(clusters_out)
    ])
    
    # 4. Silver Model Generation
    # Input: clusters.json
    model_out = data_dir / "silver_model_draft.json"
    
    run_step("4. Silver Model Generation", [
        python_exe, str(src_dir / "ai" / "silver_model_generator.py"),
        "-i", str(clusters_out),
        "-o", str(model_out)
    ])
    
    # 5. Generate Snowflake SQL
    # Input: silver_model.json
    sql_out = data_dir / "silver_model_DDL.sql"
    db_name = os.getenv("SNOWFLAKE_DATABASE", "PROD_C360_DB")
    
    run_step("5. Generate Snowflake DDL", [
        python_exe, str(src_dir / "ai" / "generate_silver_snowflake_ddl.py"),
        "-i", str(model_out),
        "-o", str(sql_out),
        "--db", db_name,
        "--schema", "SILVER",
        "--transient",
        "--include-comments"
    ])
    
    print(f"\n🎉 Pipeline Finished Successfully!")
    print(f"Final DDL Script: {sql_out}")

if __name__ == "__main__":
    main()
