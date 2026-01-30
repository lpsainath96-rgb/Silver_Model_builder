# CBRE AI Silver Model Builder

An AI-powered data modeling pipeline that automatically transforms Bronze (raw) layer data models into harmonized Silver layer models for CBRE's Non-GWS Revenue analytics. The pipeline uses **Snowflake Cortex AI** functions for semantic analysis, clustering, and model generation to automate the typically manual process of creating standardized Silver layer schemas.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Module Documentation](#module-documentation)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project automates the transformation of heterogeneous Bronze layer data models into a standardized Silver layer schema. The pipeline:

1. **Extracts** metadata from Snowflake Bronze DDL files
2. **Generates** synthetic data for profiling (optional) or **Profiles** real Snowflake data
3. **Creates** business-friendly column descriptions using `AI_COMPLETE`
4. **Clusters** columns by semantic similarity using `AI_EMBED`
5. **Generates** harmonized Silver model DDL using `AI_COMPLETE`
6. **Validates** the output against reference models and computes metrics

### Key Features

- 🧠 **AI-Powered**: Uses Snowflake Cortex (`AI_COMPLETE`, `AI_EMBED`) for intelligent column clustering and model generation
- 🔄 **Graceful Fallbacks**: Falls back to TF-IDF embeddings and template-based descriptions if APIs fail
- 📊 **Comprehensive Validation**: Includes metrics and comparison tools
- 🎯 **Snowflake-Specific**: Optimized for Snowflake data types and syntax
- 🔧 **Modular Design**: Each stage is independently executable

## 🏗️ Architecture

The pipeline consists of four main modules:

```
src/
├── extract/          # Data extraction and profiling
├── profiling/        # Description generation
├── ai/               # AI-powered clustering and model generation
└── validation/       # Metrics and comparison tools
```

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Stages                           │
└─────────────────────────────────────────────────────────────────┘

1. BRONZE DATA
   ↓
2. Real Data Profiling (profile_real_data.py)
   → column_profiles_real.json
   ↓
3. Long Descriptions (generate_long_descriptions.py)
   → column_long_descriptions_llm.json
   ↓
4. Semantic Clustering (semantic_clustering.py)
   → semantic_clusters_llm.json
   ↓
5. Silver Model Generation (silver_model_generator.py)
   → silver_model_draft_llm.json
   ↓
6. DDL Generation (generate_silver_snowflake_ddl.py)
   → silver_model_draft_llm_snowflake.sql
   ↓
7. Validation & Metrics (validation/*.py)
   → bronze_to_silver_metrics.json
```

## 📦 Prerequisites

- **Python 3.8+**
- **Snowflake Account** with access to Cortex functions:
  - `SNOWFLAKE.CORTEX.COMPLETE` (e.g., model `llama3-70b`)
  - `SNOWFLAKE.CORTEX.EMBED_TEXT_768` (e.g., model `e5-base-v2`)
- **Environment Variables** (see Configuration section)

## 🔧 Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (see Configuration section below).

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `src/` directory or set environment variables:

```bash
# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=your_role
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# Snowflake AI Configuration (Optional defaults shown)
SNOWFLAKE_LLM_MODEL=llama3-70b
SNOWFLAKE_EMBED_MODEL=e5-base-v2
```

### Directory Structure

The pipeline expects the following directory structure:

```
CBRE_AI_Silver_POC/
├── data/
│   ├── V4-Demo/                    # Current data location
│   │   ├── NGR-B DDL.txt           # Input Bronze DDL
│   │   ├── bronze_NGR-B_metadata.json
│   │   └── profile/                # Profiling outputs
├── src/
│   ├── extract/
│   ├── profiling/
│   ├── ai/
│   └── validation/
└── requirements.txt
```

## 🚀 Usage

### Quick Start - Full Pipeline

Run each stage sequentially:

```bash
# 1. Profile Real Data (NEW)
python src/extract/profile_real_data.py \
  -o data/V4-Demo/profile/column_profiles.json

# 2. Generate long descriptions (LLM mode)
python src/profiling/generate_long_descriptions.py \
  --mode llm \
  -p data/V4-Demo/profile/column_profiles.json \
  -o data/V4-Demo/profile/column_long_descriptions_llm.json

# 3. Semantic clustering
python src/ai/semantic_clustering.py \
  -d data/V4-Demo/profile/column_long_descriptions_llm.json \
  -o data/V4-Demo/profile/semantic_clusters_llm.json

# 4. Generate Silver model
python src/ai/silver_model_generator.py
# Output: data/silver_model_draft_llm.json

# 5. Generate Snowflake DDL
python src/ai/generate_silver_snowflake_ddl.py \
  -i data/V4-Demo/silver_model_draft_llm.json \
  -o data/V4-Demo/silver_model_draft_llm_snowflake.sql \
  --transient

# 6. Compute metrics
python src/validation/bronze_to_silver_metrics.py
```

## 📄 Module Documentation

### Extract Module (`src/extract/`)

#### `profile_real_data.py` (NEW)
Profiles columns directly from Snowflake tables using SQL aggregations.

**Usage**:
```bash
python src/extract/profile_real_data.py -o profiles_output.json
```

**Output**: JSON with column statistics (Total rows, nulls, distincts, min/max).

### AI Module (`src/ai/`)

#### `semantic_clustering.py`
Clusters columns by semantic similarity using **Snowflake `AI_EMBED`**.

**Usage**:
```bash
python src/ai/semantic_clustering.py \
  -d descriptions.json \
  -o clusters_output.json
```

#### `silver_model_generator.py`
Generates Silver model JSON using **Snowflake `AI_COMPLETE`**.

**Usage**:
```bash
python src/ai/silver_model_generator.py
```

## 🔍 Troubleshooting

### Snowflake Cortex Errors
**Issue**: `Snowflake Cortex error: ...`
**Solutions**:
- Verify your snowflake account region supports Cortex.
- Ensure your role has `CORTEX_USER` database role or equivalent privileges.
- Check if the model name (`llama3-70b`, `e5-base-v2`) is available in your region.

### Common Connection Issues
**Issue**: Cannot connect to Snowflake.
**Solutions**:
- Verify all `SNOWFLAKE_*` environment variables.
- Remove `.snowflakecomputing.com` suffix from `SNOWFLAKE_ACCOUNT` if present.

## 📝 Notes

- **Cost Awareness**: Snowflake Cortex functions consume Snowflake credits.
- **Security**: Ensure `.env` files are in `.gitignore`.
