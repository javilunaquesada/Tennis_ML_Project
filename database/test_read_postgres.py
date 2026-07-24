from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import engine

import pandas as pd
from src.db import engine

query = """
SELECT *
FROM matches
LIMIT 5
"""

df = pd.read_sql(query, engine)

print(df.head())