"""
Unified Profiling & Visualization Script

This script runs the entire profiling pipeline in one go:
1. Connects to Snowflake to profile real data (profile_real_data.py)
2. Generates visualization charts from the results (profile_visualizer.py)
"""
import subprocess
import sys
from pathlib import Path

def run_pipeline():
    base_dir = Path(__file__).resolve().parent.parent
    src_dir = base_dir / "src"
    
    print("🚀 Starting Unified Profiling Pipeline...")
    
    # Step 1: Profile Data
    print("\n[Step 1/2] Profiling Real Data from Snowflake...")
    profile_script = src_dir / "extract" / "profile_real_data.py"
    try:
        subprocess.run([sys.executable, str(profile_script)], check=True)
        print("✅ Profiling complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Profiling failed with exit code {e.returncode}")
        return

    # Step 2: Visualize Data
    print("\n[Step 2/2] Generating Visualization Charts...")
    viz_script = src_dir / "profile_visualizer.py"
    try:
        subprocess.run([sys.executable, str(viz_script)], check=True)
        print("✅ Visualization complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Visualization failed with exit code {e.returncode}")
        return

    print("\n🎉 Pipeline Finished Successfully!")
    print(f"Check outputs in: {base_dir / 'data' / 'profile'}")

if __name__ == "__main__":
    run_pipeline()
