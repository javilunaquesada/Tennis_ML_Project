# database/test_connection.py

from sqlalchemy import create_engine
from urllib.parse import quote_plus

password = quote_plus("0106")

engine = create_engine(
    f"postgresql://postgres:{password}@localhost:5432/tennis_ml"
)

with engine.connect() as conn:
    print("Connection successful!")