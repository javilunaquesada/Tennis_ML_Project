import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

from src import db

load_dotenv()

class DatabaseManager:
    """
    Central class responsible for all database interactions.
    """

    def __init__(self):
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """
        Creates SQLAlchemy engine using environment variables.
        """
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        name = os.getenv("POSTGRES_DB")

        database_url = (
            f"postgresql://{user}:{password}"
            f"@{host}:{port}/{db}"
        )

        return create_engine(database_url)
    
    def get_matches(self):
        """
        Load all matches from PostgreSQL.
        """
        query = """
        SELECT *
        FROM matches
        """

        matches = pd.read_sql(query, self.engine)

        matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])

        return matches