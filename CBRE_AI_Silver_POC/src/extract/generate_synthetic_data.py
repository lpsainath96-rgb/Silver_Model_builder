import json
import argparse
import random
import string
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

# Heuristic vocab pools
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
COUNTRIES = ["US","UK","DE","FR","IN","AU","CA","BR","JP"]
CURRENCIES = ["USD","EUR","GBP","INR","AUD","CAD"]
BUSINESS_UNITS = ["Advisory","Valuation","CapitalMarkets","PropertyMgmt","Workplace"]
LOB_L3 = ["GWS","Advisory","Digital","Investments"]
FUNCTIONAL_UNITS = ["Finance","HR","IT","Ops","Analytics"]
MANAGING_OFFICES = ["NewYork","London","Berlin","Delhi","Sydney"]

DEFAULT_ROWS = 100

def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_timestamp(days_back=365):
    base = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return base.replace(microsecond=0).isoformat(sep=' ')  # Snowflake TIMESTAMP_NTZ format compatible

def infer_category(col_name: str):
    n = col_name.lower()
    if "month_name" in n: return "month"
    if n == "year": return "year"
    if "currency" in n: return "currency"
    if "country" in n: return "country"
    if "business_unit" in n: return "business_unit"
    if "reporting_lob_l3" in n: return "lob_l3"
    if "functional_unit" in n or "function_unit" in n: return "functional_unit"
    if "managing_office" in n: return "managing_office"
    if "md5_hash" in n: return "md5"
    if "value" in n: return "numeric_value"
    if "sf_insert_dt" in n: return "timestamp"
    return "generic"

def generate_value(col_name: str, datatype: str, row_index: int):
    cat = infer_category(col_name)
    if cat == "month":
        return MONTHS[row_index % len(MONTHS)]
    if cat == "year":
        return random.randint(2019, 2025)
    if cat == "currency":
        return random.choice(CURRENCIES)
    if cat == "country":
        return random.choice(COUNTRIES)
    if cat == "business_unit":
        return random.choice(BUSINESS_UNITS)
    if cat == "lob_l3":
        return random.choice(LOB_L3)
    if cat == "functional_unit":
        return random.choice(FUNCTIONAL_UNITS)
    if cat == "managing_office":
        return random.choice(MANAGING_OFFICES)
    if cat == "timestamp":
        return random_timestamp()
    if cat == "numeric_value":
        # differentiate GR/NR/SOP margins with ranges
        n = col_name.lower()
        base = random.uniform(1000, 100000)
        if "margin" in n:
            base = random.uniform(0, 50000)
        if "usd_value" in n and "sop" in n:
            base = random.uniform(1000, 75000)
        # 2 decimal places for NUMBER(38,2)
        return round(base, 2)
    if cat == "md5":
        # placeholder hash of name+index
        return hashlib.md5(f"{col_name}:{row_index}:{random.random()}".encode()).hexdigest()
    # generic fallback by datatype
    if datatype.startswith("NUMBER"):
        if "," in datatype:  # NUMBER(38,2)
            return round(random.uniform(1000, 90000), 2)
        return random.randint(1, 100000)
    if datatype.startswith("TIMESTAMP"):
        return random_timestamp()
    if datatype.startswith("VARCHAR"):
        return random_string(12)
    return random_string(8)

def generate_table(table_name: str, columns: list, rows: int):
    # Build CSV lines
    headers = [c["name"] for c in columns]
    lines = [",".join(headers)]
    for r in range(rows):
        row_vals = []
        for col in columns:
            val = generate_value(col["name"], col["datatype"], r)
            # quote if contains non-alnum or is string
            if isinstance(val, str):
                # Escape quotes
                safe = val.replace('"','""')
                row_vals.append(f'"{safe}"')
            else:
                row_vals.append(str(val))
        lines.append(",".join(row_vals))
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CSV data from Bronze metadata JSON")
    script_dir = Path(__file__).resolve().parent
    # Ascend to project root (CBRE_AI_SilverPOC) not src
    project_root = script_dir.parents[1]
    default_meta = project_root / "data" / "V2" / "bronze_NGR-B_metadata.json"
    default_out_dir = project_root / "data" / "V2" / "synthetic"
    parser.add_argument("-m","--metadata", default=str(default_meta), help="Path to metadata JSON")
    parser.add_argument("-o","--outdir", default=str(default_out_dir), help="Output directory for synthetic CSVs")
    parser.add_argument("-r","--rows", type=int, default=DEFAULT_ROWS, help="Rows per table (15-20 recommended)")
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for table_name, info in meta.items():
        cols = info.get("columns", [])
        if not cols:
            continue
        csv_content = generate_table(table_name, cols, args.rows)
        # file name simplified
        safe_table = table_name.split('.')[-1]
        file_path = out_dir / f"{safe_table}.csv"
        file_path.write_text(csv_content, encoding="utf-8")
        print(f"Generated synthetic data: {file_path}")

    print(f"Synthetic data generation complete. Files located in: {out_dir}")

if __name__ == "__main__":
    main()
