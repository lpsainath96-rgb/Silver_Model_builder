"""Compute extended Bronze -> Silver model metrics.

Inputs:
  - Bronze metadata JSON (columns across source tables)
  - Final Silver Snowflake DDL (fact + dims)
Outputs:
  JSON with:
    bronze_total_columns
    silver_total_columns
    bronze_unique_tokens
    silver_unique_tokens
    token_reuse_ratio (silver_unique_tokens / bronze_unique_tokens)
    silver_column_coverage_vs_bronze (% of bronze column names appearing directly or fuzzily in silver)
    hierarchical_depth (max reporting_division / reporting_lob level found)
    measure_columns (list)
    measure_count
    dimension_count
    measure_dimension_ratio
    precision_preserved_count (NUMBER(x,y) kept vs flattened)
    timestamp_columns
    potential_surrogate_key_candidates (columns ending in _ID or hash)
    unmapped_bronze_columns
"""
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

BRONZE_PATH = Path(__file__).resolve().parents[2] / 'data' / 'V2' / 'bronze_NGR-B_metadata.json'
SILVER_DDL_PATH = Path(__file__).resolve().parents[2] / 'data' / 'V2' / 'silver_model_non_gws_revenue_snowflake.sql'
OUT_PATH = Path(__file__).resolve().parents[2] / 'data' / 'V2' / 'bronze_to_silver_extended_metrics.json'

NUMBER_PATTERN = re.compile(r'NUMBER\((\d+),(\d+)\)', re.IGNORECASE)
COLUMN_LINE = re.compile(r'"?([A-Z0-9_]+)"?\s+([A-Z0-9_()]+)')

MEASURE_KEYWORDS = ("REVENUE","PROFIT","MARGIN","VALUE","USD","LOCAL","SOP")


def fuzzy_key(s: str) -> str:
    return s.lower().replace('_', '')

def load_bronze_cols(data: dict):
    cols = []
    for tbl, info in data.items():
        for col in info.get('columns', []):
            cols.append(col['name'].upper())
    return cols

def parse_silver_columns(sql_text: str):
    cols = []
    for line in sql_text.splitlines():
        line = line.strip().rstrip(',')
        m = COLUMN_LINE.match(line)
        if m:
            cols.append((m.group(1).upper(), m.group(2).upper()))
    return cols

def classify_measure(col: str) -> bool:
    u = col.upper()
    return any(k in u for k in MEASURE_KEYWORDS)

def main():
    bronze = json.loads(BRONZE_PATH.read_text())
    silver_sql = SILVER_DDL_PATH.read_text()
    bronze_cols = load_bronze_cols(bronze)
    silver_cols = parse_silver_columns(silver_sql)

    silver_names = [c for c,_ in silver_cols]

    bronze_set = set(bronze_cols)
    silver_set = set(silver_names)

    overlaps = bronze_set & silver_set

    # Fuzzy matches (underscore/case-insensitive collapse)
    silver_fuzzy = {fuzzy_key(c): c for c in silver_set}
    fuzzy_matches = []
    for b in bronze_set - overlaps:
        fk = fuzzy_key(b)
        if fk in silver_fuzzy:
            fuzzy_matches.append((b, silver_fuzzy[fk]))

    coverage_pct = round((len(overlaps) + len(fuzzy_matches)) / len(bronze_set) * 100, 2) if bronze_set else 0

    # Token analysis
    bronze_tokens = []
    for c in bronze_cols:
        bronze_tokens.extend([t for t in c.lower().split('_') if t])
    silver_tokens = []
    for c in silver_names:
        silver_tokens.extend([t for t in c.lower().split('_') if t])
    bronze_unique_tokens = set(bronze_tokens)
    silver_unique_tokens = set(silver_tokens)
    token_reuse_ratio = round(len(silver_unique_tokens) / (len(bronze_unique_tokens) or 1), 3)

    # Hierarchical depth (reporting_division / reporting_lob max level number)
    div_levels = []
    lob_levels = []
    for name in silver_names:
        m_div = re.search(r'REPORTING_DIVISION_L(\d+)', name)
        if m_div:
            div_levels.append(int(m_div.group(1)))
        m_lob = re.search(r'REPORTING_LOB_L(\d+)', name)
        if m_lob:
            lob_levels.append(int(m_lob.group(1)))
    hierarchical_depth = {
        'division_max_level': max(div_levels) if div_levels else 0,
        'lob_max_level': max(lob_levels) if lob_levels else 0
    }

    measure_cols = [c for c in silver_names if classify_measure(c)]
    dimension_cols = [c for c in silver_names if c not in measure_cols]
    measure_dimension_ratio = round(len(measure_cols) / (len(dimension_cols) or 1), 3)

    # Precision preservation counts
    precision_preserved = [t for _, t in silver_cols if NUMBER_PATTERN.search(t)]
    precision_preserved_count = len(precision_preserved)

    timestamp_columns = [c for c,t in silver_cols if t.startswith('TIMESTAMP') or t.startswith('DATE')]

    surrogate_candidates = [c for c in silver_names if c.endswith('_ID') or 'HASH' in c]

    unmapped_bronze = sorted(list(bronze_set - overlaps - {b for b,_ in fuzzy_matches}))

    out = {
        'bronze_total_columns': len(bronze_set),
        'silver_total_columns': len(silver_set),
        'bronze_unique_tokens': len(bronze_unique_tokens),
        'silver_unique_tokens': len(silver_unique_tokens),
        'token_reuse_ratio': token_reuse_ratio,
        'silver_column_coverage_vs_bronze_%': coverage_pct,
        'hierarchical_depth': hierarchical_depth,
        'measure_columns': measure_cols,
        'measure_count': len(measure_cols),
        'dimension_count': len(dimension_cols),
        'measure_dimension_ratio': measure_dimension_ratio,
        'precision_preserved_count': precision_preserved_count,
        'timestamp_columns': timestamp_columns,
        'potential_surrogate_key_candidates': surrogate_candidates,
        'direct_overlaps': sorted(list(overlaps)),
        'fuzzy_matches': fuzzy_matches,
        'unmapped_bronze_columns': unmapped_bronze
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Bronze->Silver extended metrics written -> {OUT_PATH}")

if __name__ == '__main__':
    main()
