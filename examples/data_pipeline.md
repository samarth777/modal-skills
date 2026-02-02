# Data Processing Pipeline Example

A complete example of a scalable data processing pipeline on Modal.

```python
import modal
from datetime import datetime

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas==2.2.0",
        "pyarrow==15.0.0",
        "polars==0.20.0",
        "duckdb==0.10.0",
        "boto3",
    )
)

app = modal.App("data-pipeline", image=image)

# --- Volumes ---
raw_data = modal.Volume.from_name("raw-data", create_if_missing=True)
processed_data = modal.Volume.from_name("processed-data", create_if_missing=True)

RAW_PATH = "/raw"
PROCESSED_PATH = "/processed"

# --- Extract Stage ---
@app.function(
    volumes={RAW_PATH: raw_data},
    secrets=[modal.Secret.from_name("aws-credentials")],
    timeout=3600,
    memory=8192,
)
def extract_from_s3(
    bucket: str,
    prefix: str,
    date: str,
) -> list[str]:
    """Download files from S3 to Modal volume."""
    import boto3
    import os
    
    s3 = boto3.client("s3")
    
    # List objects
    response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{date}")
    files = [obj["Key"] for obj in response.get("Contents", [])]
    
    downloaded = []
    for key in files:
        local_path = f"{RAW_PATH}/{key}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        downloaded.append(local_path)
    
    raw_data.commit()
    return downloaded

# --- Transform Stage ---
@app.function(
    volumes={
        RAW_PATH: raw_data,
        PROCESSED_PATH: processed_data,
    },
    memory=16384,
    cpu=4,
)
def transform_file(input_path: str) -> str:
    """Transform a single file using Polars."""
    import polars as pl
    import os
    
    # Read raw data
    df = pl.read_parquet(input_path)
    
    # Apply transformations
    df = (
        df
        .filter(pl.col("status") == "active")
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("amount").cast(pl.Float64),
            (pl.col("amount") * pl.col("quantity")).alias("total"),
        ])
        .drop_nulls()
    )
    
    # Write output
    output_path = input_path.replace(RAW_PATH, PROCESSED_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_parquet(output_path)
    
    processed_data.commit()
    return output_path

# --- Aggregate Stage ---
@app.function(
    volumes={PROCESSED_PATH: processed_data},
    memory=32768,
    cpu=8,
)
def aggregate_data(file_paths: list[str]) -> dict:
    """Aggregate processed files using DuckDB."""
    import duckdb
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    # Register all files as a view
    file_pattern = f"{PROCESSED_PATH}/**/*.parquet"
    
    # Run aggregation query
    result = con.execute(f"""
        SELECT
            date_trunc('day', timestamp) as date,
            COUNT(*) as count,
            SUM(total) as total_amount,
            AVG(total) as avg_amount
        FROM read_parquet('{file_pattern}')
        GROUP BY 1
        ORDER BY 1
    """).fetchdf()
    
    # Save summary
    summary_path = f"{PROCESSED_PATH}/summary/daily_summary.parquet"
    result.to_parquet(summary_path)
    processed_data.commit()
    
    return {
        "rows_processed": int(result["count"].sum()),
        "total_amount": float(result["total_amount"].sum()),
        "date_range": [
            result["date"].min().isoformat(),
            result["date"].max().isoformat(),
        ],
    }

# --- Load Stage ---
@app.function(
    volumes={PROCESSED_PATH: processed_data},
    secrets=[modal.Secret.from_name("database-credentials")],
)
def load_to_database(summary_path: str) -> int:
    """Load summary data to database."""
    import pandas as pd
    import os
    # from sqlalchemy import create_engine
    
    df = pd.read_parquet(summary_path)
    
    # Load to database
    # engine = create_engine(os.environ["DATABASE_URL"])
    # df.to_sql("daily_summary", engine, if_exists="append", index=False)
    
    print(f"Loaded {len(df)} rows to database")
    return len(df)

# --- Pipeline Orchestrator ---
@app.function(timeout=7200)
def run_pipeline(
    bucket: str,
    prefix: str,
    date: str,
) -> dict:
    """Run the full ETL pipeline."""
    from datetime import datetime
    
    start_time = datetime.now()
    
    # Extract
    print("Starting extraction...")
    raw_files = extract_from_s3.remote(bucket, prefix, date)
    print(f"Extracted {len(raw_files)} files")
    
    # Transform in parallel
    print("Starting transformation...")
    processed_files = list(transform_file.map(raw_files))
    print(f"Transformed {len(processed_files)} files")
    
    # Aggregate
    print("Starting aggregation...")
    summary = aggregate_data.remote(processed_files)
    print(f"Aggregation complete: {summary}")
    
    # Load
    print("Loading to database...")
    summary_path = f"{PROCESSED_PATH}/summary/daily_summary.parquet"
    rows_loaded = load_to_database.remote(summary_path)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "success",
        "files_processed": len(raw_files),
        "rows_loaded": rows_loaded,
        "summary": summary,
        "duration_seconds": duration,
    }

# --- Scheduled Job ---
@app.function(schedule=modal.Cron("0 6 * * *"))  # 6 AM daily
def daily_pipeline():
    """Run pipeline daily for yesterday's data."""
    from datetime import datetime, timedelta
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    result = run_pipeline.remote(
        bucket="my-data-bucket",
        prefix="events",
        date=yesterday,
    )
    
    print(f"Daily pipeline complete: {result}")
    return result

# --- Web Trigger ---
@app.function()
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def trigger_pipeline(body: dict) -> dict:
    """Trigger pipeline via API."""
    call = run_pipeline.spawn(
        bucket=body["bucket"],
        prefix=body["prefix"],
        date=body["date"],
    )
    
    return {"call_id": call.object_id, "status": "started"}

@app.function()
@modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
def get_pipeline_status(call_id: str) -> dict:
    """Check pipeline status."""
    call = modal.FunctionCall.from_id(call_id)
    
    try:
        result = call.get(timeout=0)
        return {"status": "completed", "result": result}
    except TimeoutError:
        return {"status": "running"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# --- CLI ---
@app.local_entrypoint()
def main(
    bucket: str = "my-data-bucket",
    prefix: str = "events",
    date: str = "2024-01-01",
):
    result = run_pipeline.remote(bucket, prefix, date)
    print(f"Pipeline complete: {result}")
```

## Usage

```bash
# Run pipeline manually
modal run data_pipeline.py --bucket my-bucket --prefix events --date 2024-01-15

# Deploy with scheduled job
modal deploy data_pipeline.py

# Trigger via API
curl -X POST https://your-workspace--data-pipeline-trigger-pipeline.modal.run \
  -H "Modal-Key: $TOKEN_ID" \
  -H "Modal-Secret: $TOKEN_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"bucket": "my-bucket", "prefix": "events", "date": "2024-01-15"}'
```
