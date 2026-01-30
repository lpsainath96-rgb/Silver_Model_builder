import re
import json
import argparse
from pathlib import Path

def parse_ddl(ddl_text: str):
    tables = {}
    current_table = None
    inside_table = False
    for line in ddl_text.splitlines():
        line = line.strip()
        # CREATE / TRANSIENT / OR REPLACE TABLE
        if re.search(r'CREATE\s+(OR\s+REPLACE\s+)?(TRANSIENT\s+)?TABLE', line, re.I):
            match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:TRANSIENT\s+)?TABLE\s+([^\s(]+)', line, re.I)
            if match:
                current_table = match.group(1)
                tables[current_table] = {"columns": []}
                inside_table = True
            continue
        # End of table
        if inside_table and line.startswith(")"):
            inside_table = False
            current_table = None
            continue
        # Columns
        if inside_table and current_table and line and not line.upper().startswith(("PRIMARY","FOREIGN","CONSTRAINT")):
            # remove trailing comma
            cleaned = line.rstrip(",")
            parts = re.split(r'\s+', cleaned)
            if len(parts) >= 2:
                col_name = parts[0].strip('"')
                col_type = parts[1].upper()
                tables[current_table]["columns"].append({"name": col_name, "datatype": col_type})
    return tables

def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_in = script_dir.parent / "data" / "V2" / "NGR-B DDL.txt"
    default_out = script_dir.parent / "data" / "V2" / "bronze_NGR-B_metadata.json"
    parser.add_argument("-i","--input", default=str(default_in), help="Path to Bronze DDL text file")
    parser.add_argument("-o","--output", default=str(default_out), help="Output metadata JSON path")
    args = parser.parse_args()

    ddl_path = Path(args.input)
    if not ddl_path.exists():
        raise FileNotFoundError(f"Input DDL not found: {ddl_path}")

    ddl_text = ddl_path.read_text(encoding="utf-8")
    metadata = parse_ddl(ddl_text)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2))
    print(f"Extracted metadata saved to: {out_path}")

if __name__ == "__main__":
    main()