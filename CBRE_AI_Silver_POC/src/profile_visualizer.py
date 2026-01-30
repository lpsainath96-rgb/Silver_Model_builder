"""
Profile Visualizer

Reads the JSON output from profile_real_data.py and generates charts.
does NOT connect to any database.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_profiles(json_path: str, output_dir: str):
    path = Path(json_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"Reading profiles from: {path}")
    data = json.loads(path.read_text())
    
    # Prepare output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Nulls Analysis (Bar Chart)
    # We aggregate nulls across all tables/columns to find the "dirtiest" columns
    top_null_columns = []
    
    for table_name, columns in data.items():
        for col_name, stats in columns.items():
            nulls = stats.get('nulls', 0)
            total = stats.get('total', 0)
            if total > 0:
                pct_null = (nulls / total) * 100
                top_null_columns.append((f"{table_name}.{col_name}", pct_null))
    
    # Sort and take top 20
    top_null_columns.sort(key=lambda x: x[1], reverse=True)
    top_20 = top_null_columns[:20]
    
    if top_20:
        plt.figure(figsize=(12, 8))
        names = [x[0] for x in top_20]
        values = [x[1] for x in top_20]
        
        plt.barh(names, values, color='salmon')
        plt.xlabel('% Null Values')
        plt.title('Top 20 Columns with Highest Null Percentage')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        outfile = out_dir / "null_analysis.png"
        plt.savefig(outfile)
        print(f"✅ Generated chart: {outfile}")
        plt.close()
    else:
        print("ℹ️ No nulls found to chart.")

    # 2. Distinct Values (Box Plot or Histogram equivalent)
    # Let's count datatypes
    datatypes = {}
    for table_name, columns in data.items():
        for col_name, stats in columns.items():
            dtype = stats.get('datatype', 'UNKNOWN')
            # Simplify snowflake types (e.g. NUMBER(38,0) -> NUMBER)
            simple_dtype = dtype.split('(')[0]
            datatypes[simple_dtype] = datatypes.get(simple_dtype, 0) + 1
            
    if datatypes:
        plt.figure(figsize=(10, 6))
        plt.pie(datatypes.values(), labels=datatypes.keys(), autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Column Data Types')
        
        outfile = out_dir / "datatype_distribution.png"
        plt.savefig(outfile)
        print(f"✅ Generated chart: {outfile}")
        plt.close()

if __name__ == "__main__":
    # Default paths based on project structure
    base = Path(__file__).resolve().parents[1]
    input_json = base / "data" / "profile" / "column_profiles_real.json"
    output_charts = base / "data" / "profile" / "charts"
    
    visualize_profiles(input_json, output_charts)
