import csv
import json
import argparse
from pathlib import Path
from collections import Counter

NUMERIC_PREFIXES = ("NUMBER", "FLOAT", "INT", "DEC", "DOUBLE")
TIMESTAMP_PREFIXES = ("TIMESTAMP",)


def is_numeric(datatype: str) -> bool:
    return datatype.upper().startswith(NUMERIC_PREFIXES) or datatype.upper().startswith("USD_VALUE")


def load_metadata(meta_path: Path):
    return json.loads(meta_path.read_text())


def scan_csv(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def profile_column(values):
    total = len(values)
    nulls = sum(1 for v in values if v == '' or v is None)
    non_null_values = [v for v in values if v != '' and v is not None]
    distinct = len(set(non_null_values))
    freq = Counter(non_null_values)
    top3 = [val for val, _ in freq.most_common(3)]
    samples = non_null_values[:2] if len(non_null_values) >= 2 else non_null_values
    return {
        "total": total,
        "nulls": nulls,
        "distinct": distinct,
        "top_values": top3,
        "sample_values": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile synthetic CSV data")
    script_dir = Path(__file__).resolve().parent
    default_meta = script_dir.parent / "data" / "V2" / "bronze_NGR-B_metadata.json"
    default_synth = script_dir.parent / "data" / "V2" / "synthetic"
    default_out = script_dir.parent / "data" / "V2" / "profile" / "column_profiles.json"

    parser.add_argument("-m", "--metadata", default=str(default_meta), help="Metadata JSON path")
    parser.add_argument("-s", "--synthetic", default=str(default_synth), help="Directory of synthetic CSV files")
    parser.add_argument("-o", "--output", default=str(default_out), help="Output JSON path for profiles")
    args = parser.parse_args()

    meta = load_metadata(Path(args.metadata))
    synth_dir = Path(args.synthetic)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = {}

    # Build datatype lookup
    datatype_lookup = {}
    for table, info in meta.items():
        for col in info.get("columns", []):
            datatype_lookup[(table.split('.')[-1], col["name"])] = col["datatype"]

    for csv_file in synth_dir.glob("*.csv"):
        table_key = csv_file.stem  # expects last part of table name
        header, rows = scan_csv(csv_file)
        if not header:
            continue
        cols_values = {h: [] for h in header}
        for r in rows:
            for i, h in enumerate(header):
                cols_values[h].append(r[i])

        table_profile = {}
        for col_name, vals in cols_values.items():
            base_prof = profile_column(vals)
            datatype = datatype_lookup.get((table_key, col_name), "VARCHAR")
            # Numeric range if numeric
            if is_numeric(datatype):
                numeric_vals = []
                for v in vals:
                    try:
                        numeric_vals.append(float(v))
                    except ValueError:
                        pass
                if numeric_vals:
                    base_prof["range"] = {
                        "min": min(numeric_vals),
                        "max": max(numeric_vals)
                    }
            table_profile[col_name] = {
                "datatype": datatype,
                **base_prof
            }
        profiles[table_key] = table_profile
        print(f"Profiled table: {table_key}")

    out_path.write_text(json.dumps(profiles, indent=2))
    print(f"Column profiles written to: {out_path}")

if __name__ == "__main__":
    main()
