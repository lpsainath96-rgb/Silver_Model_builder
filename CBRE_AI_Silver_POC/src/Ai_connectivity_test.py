"""Snowflake Cortex Connectivity Test.

Run this script to verify that your environment variables are set correctly
and that you can successfully call Snowflake AI functions.
"""

import os
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from specific location or default
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)
print(f"Loading env from: {env_path}")

def get_snowflake_connection():
    """Create a Snowflake connection."""
    account = os.getenv("SNOWFLAKE_ACCOUNT", "")
    if account.endswith(".snowflakecomputing.com"):
        account = account.replace(".snowflakecomputing.com", "")
    
    # Try normalized account name (underscores to hyphens)
    normalized_account = account.replace("_", "-")
    print(f"Original Account: {account}")
    print(f"Normalized Account: {normalized_account}")
    
    # Try with both variations
    for acc in [account, normalized_account]:
        print(f"\n--- Attempting connection with account: {acc} ---")
        try:
            return snowflake.connector.connect(
                account=acc,
                user=os.getenv("SNOWFLAKE_USER"),
                password=os.getenv("SNOWFLAKE_PASSWORD"),
                role=os.getenv("SNOWFLAKE_ROLE"),
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
                database=os.getenv("SNOWFLAKE_DATABASE"),
                schema=os.getenv("SNOWFLAKE_SCHEMA"),
            )
        except Exception as e:
            print(f"Connection failed for {acc}: {e}")
            
    print("\nAll connection attempts failed. Trying insecure_mode with normalized account...")
    return snowflake.connector.connect(
        account=normalized_account,
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        insecure_mode=True
    )

def test_cortex_functions():
    try:
        conn = get_snowflake_connection()
        print("\n✅ Snowflake Connection Successful!")
        
        # Test AI_COMPLETE
        llm_model = os.getenv("SNOWFLAKE_LLM_MODEL", "llama3-70b")
        print(f"\n[Test 1] Testing AI_COMPLETE with model '{llm_model}'...")
        prompt = "Explain in one sentence why data engineering is important."
        sql_complete = f"SELECT AI_COMPLETE('{llm_model}', '{prompt}')"
        
        with conn.cursor() as cur:
            cur.execute(sql_complete)
            result = cur.fetchone()[0]
            print(f"Response: {result}")
            
        print("✅ AI_COMPLETE Test Passed!")

        # Test AI_EMBED
        embed_model = os.getenv("SNOWFLAKE_EMBED_MODEL", "e5-base-v2")
        print(f"\n[Test 2] Testing AI_EMBED with model '{embed_model}'...")
        text = "Semantic search embedding test."
        sql_embed = f"SELECT AI_EMBED('{embed_model}', '{text}')"
        
        with conn.cursor() as cur:
            cur.execute(sql_embed)
            result = cur.fetchone()[0]
            # Result is a JSON string or list representing the vector
            print(f"Response (truncated): {str(result)[:50]}...")
            
        print("✅ AI_EMBED Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Check your .env file credentials.")
        print("2. Ensure your Snowflake Account and Region support Cortex functions.")
        print("3. Verify your Role has access to the Database/Schema and Cortex functions.")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_cortex_functions()