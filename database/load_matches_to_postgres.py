from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Database connection
# -----------------------------

database_url = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}?sslmode=require"
)

engine = create_engine(database_url)

# -----------------------------
# Load CSV
# -----------------------------

PROJECT_ROOT = Path(__file__).parent.parent

csv_path = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "matches_with_global_elo.csv"
)

matches = pd.read_csv(csv_path)

print(f"Loaded {len(matches)} rows")

# -----------------------------
# Load into PostgreSQL
# -----------------------------

matches.to_sql(
    "matches",
    engine,
    if_exists="replace",
    index=False
)

print("Table 'matches' created successfully!")