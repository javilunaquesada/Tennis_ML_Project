from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


# -----------------------------
# Database connection
# -----------------------------

engine = create_engine(
    "postgresql://postgres:0106@localhost:5432/tennis_ml"
)

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