"""Semantic clustering using long descriptions.

Orchestrates embedding + clustering in one step for convenience.
Uses Snowflake AI_EMBED function for embeddings.
Falls back gracefully to TF-IDF if Snowflake connection fails.
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import snowflake.connector

load_dotenv()

# Snowflake connection parameters
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

# Embedding model to use with AI_EMBED
EMBED_MODEL = os.getenv("SNOWFLAKE_EMBED_MODEL", "e5-base-v2")


def get_snowflake_connection():
    """Create a Snowflake connection."""
    account = SNOWFLAKE_ACCOUNT
    if account.endswith(".snowflakecomputing.com"):
        account = account.replace(".snowflakecomputing.com", "")
    return snowflake.connector.connect(
        account=account,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )


def get_embeddings_snowflake(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Get embeddings using Snowflake AI_EMBED function.
    
    Uses batched queries to handle large numbers of texts efficiently.
    """
    vectors: List[List[float]] = []
    conn = get_snowflake_connection()
    
    try:
        with conn.cursor() as cur:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Build a query that embeds multiple texts at once
                # Using UNION ALL to combine results
                queries = []
                for idx, text in enumerate(batch):
                    # Escape single quotes in text
                    escaped_text = text.replace("'", "''")
                    queries.append(f"SELECT {idx + i} AS idx, AI_EMBED('{EMBED_MODEL}', '{escaped_text}') AS embedding")
                
                sql = " UNION ALL ".join(queries) + " ORDER BY idx"
                cur.execute(sql)
                
                for row in cur.fetchall():
                    # AI_EMBED returns an ARRAY, which comes as a list
                    embedding = row[1]
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    vectors.append(embedding)
                    
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
    finally:
        conn.close()
    
    return vectors


def fallback_embeddings(texts: List[str]) -> List[List[float]]:
    """Local TF-IDF fallback to produce vectors when Snowflake embedding fails.
    Returns dense vectors as lists of floats suitable for clustering.
    """
    tfidf = TfidfVectorizer(max_features=512)
    mat = tfidf.fit_transform(texts)
    dense = mat.toarray().tolist()
    return dense


def choose_cluster_count(n: int) -> int:
    """Heuristic: sqrt(n) rounded, min 2."""
    return max(2, int(round(math.sqrt(n))))


def cluster(embeddings: List[List[float]], labels: List[str], k: int) -> Dict[int, List[str]]:
    """Cluster embeddings using KMeans."""
    km = KMeans(n_clusters=min(k, len(labels)), random_state=42)
    y = km.fit_predict(embeddings)
    out: Dict[int, List[str]] = {}
    for lab, col in zip(y, labels):
        out.setdefault(int(lab), []).append(col)
    return out


def build_texts(descriptions_json: dict) -> List[str]:
    """Build text list from descriptions JSON."""
    all_texts = []
    for table, cols in descriptions_json.get('tables', {}).items():
        for col, desc in cols.items():
            combined = f"{table}.{col}: {desc}"
            all_texts.append(combined)
    return all_texts


def main():
    # Adjust base to project root (parents[2]) so data path resolves correctly
    base = Path(__file__).resolve().parents[2]  # project root
    default_desc = base / 'data' / 'real_run' / 'profile' / 'column_long_descriptions_llm.json'
    default_out = base / 'data' / 'real_run' / 'profile' / 'semantic_clusters.json'

    import argparse
    parser = argparse.ArgumentParser(description='Cluster columns using long descriptions embeddings')
    parser.add_argument('-d', '--descriptions', default=str(default_desc), help='Path to long descriptions JSON')
    parser.add_argument('-o', '--output', default=str(default_out), help='Output path for clusters JSON')
    parser.add_argument('-k', '--clusters', type=int, help='Force number of clusters (optional)')
    parser.add_argument('--use-tfidf', action='store_true', help='Force TF-IDF fallback instead of Snowflake AI_EMBED')
    args = parser.parse_args()

    desc_data = json.loads(Path(args.descriptions).read_text())
    texts = build_texts(desc_data)
    labels = [t.split(':', 1)[0] for t in texts]  # table.column keys
    
    print(f"[semantic_clustering] Processing {len(texts)} column descriptions...")
    
    if args.use_tfidf:
        print("[semantic_clustering] Using TF-IDF embeddings (forced via --use-tfidf)")
        embeds = fallback_embeddings(texts)
        embed_model_used = "tfidf-local"
    else:
        print(f"[semantic_clustering] Using Snowflake AI_EMBED with model '{EMBED_MODEL}'...")
        try:
            embeds = get_embeddings_snowflake(texts)
            embed_model_used = EMBED_MODEL
        except Exception as e:
            print(f"[semantic_clustering] Snowflake AI_EMBED failed ({e}); using TF-IDF fallback.")
            embeds = fallback_embeddings(texts)
            embed_model_used = "tfidf-local"
    
    k = args.clusters or choose_cluster_count(len(texts))
    print(f"[semantic_clustering] Clustering into {k} groups...")
    clusters = cluster(embeds, labels, k)
    
    payload = {
        'embed_model': embed_model_used,
        'cluster_count': len(clusters),
        'clusters': clusters
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[semantic_clustering] Clusters written to {out_path}")


if __name__ == '__main__':
    main()