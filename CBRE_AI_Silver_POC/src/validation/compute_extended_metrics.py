import json, math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2] / "data" / "V2"
BASE_COMPARISON = ROOT / "silver_ddl_comparison_baseline.json"
BASE_METRICS = ROOT / "silver_ddl_metrics_baseline.json"
PROFILES = ROOT / "profile" / "column_profiles.json"
DESCRIPTIONS = ROOT / "profile" / "column_long_descriptions_template.json"
OUT = ROOT / "silver_ddl_extended_metrics_baseline.json"

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def main():
    comparison = load(BASE_COMPARISON)
    base_metrics = load(BASE_METRICS)
    profiles = load(PROFILES)
    descriptions = load(DESCRIPTIONS)

    ref_only = comparison.get("reference_only", [])
    gen_only = comparison.get("generated_only", [])
    overlap = comparison.get("overlap", [])

    # Projected coverage if we accept all fuzzy suggestions
    fuzzy = comparison.get("fuzzy_suggestions", [])
    projected_overlap = len(overlap) + len(fuzzy)
    total_ref = len(ref_only) + len(overlap)
    projected_coverage_pct = round(projected_overlap / total_ref * 100, 2) if total_ref else 0

    # Token analytics on generated columns
    tokens = []
    for col in gen_only + overlap:
        tokens.extend([t for t in col.lower().split('_') if t])
    token_counts = Counter(tokens)
    token_uniqueness_ratio = round(len(token_counts) / (len(tokens) or 1), 3)

    # Measure vs dimension ratio (reuse earlier heuristic)
    measure_kw = ("REVENUE","PROFIT","MARGIN","VALUE","USD","LOCAL")
    def classify(c):
        u=c.upper()
        return "MEASURE" if any(k in u for k in measure_kw) else "DIMENSION"
    gen_cols_all = gen_only + overlap
    measures = [c for c in gen_cols_all if classify(c)=="MEASURE"]
    dimensions = [c for c in gen_cols_all if classify(c)=="DIMENSION"]
    measure_dimension_ratio = round(len(measures)/(len(dimensions) or 1),3)

    # Average name length
    avg_len = round(sum(len(c) for c in gen_cols_all)/(len(gen_cols_all) or 1),2)

    # Description coverage (columns with a description)
    desc_cols = descriptions.keys()
    desc_coverage_pct = round(len(set(gen_cols_all) & set(desc_cols))/ (len(gen_cols_all) or 1) * 100,2)

    # Profile availability (columns present in profiles)
    profile_tables = profiles.keys()
    profile_cols = []
    for t in profile_tables:
        for col in profiles[t].get("columns", []):
            profile_cols.append(col.get("name","").upper())
    profile_coverage_pct = round(len(set(gen_cols_all) & set(profile_cols))/ (len(gen_cols_all) or 1) * 100,2)

    # Gap closure potential (how many missing are pure naming vs truly absent measures)
    naming_gaps = len(fuzzy)
    structural_gaps = len(ref_only) - naming_gaps
    structural_gap_pct = round(structural_gaps / (len(ref_only) or 1) * 100, 2)

    extended = {
        "baseline_direct_overlap_%": base_metrics.get("direct_name_overlap_%"),
        "projected_overlap_if_fuzzy_applied_%": projected_coverage_pct,
        "measure_completeness_%": base_metrics.get("measure_completeness_%"),
        "measure_dimension_ratio": measure_dimension_ratio,
        "avg_column_name_length": avg_len,
        "token_uniqueness_ratio": token_uniqueness_ratio,
        "description_coverage_%": desc_coverage_pct,
        "profile_coverage_%": profile_coverage_pct,
        "naming_gap_count": naming_gaps,
        "structural_gap_count": structural_gaps,
        "structural_gap_pct_of_missing": structural_gap_pct,
        "generated_column_count": base_metrics.get("total_generated_columns"),
        "reference_column_count": base_metrics.get("total_reference_columns")
    }
    OUT.write_text(json.dumps(extended, indent=2))
    print(f"Extended metrics written → {OUT}")

if __name__ == "__main__":
    main()