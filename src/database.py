import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

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
            f"@{host}:{port}/{name}"
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

    def get_latest_players(self):
        """
        Build a snapshot containing the latest available information
        for every player in the database.
        """

        matches = self.get_matches()

        winner_df = (
            matches.sort_values("tourney_date")
            .groupby("winner_name")
            .tail(1)
        )

        loser_df = (
            matches.sort_values("tourney_date")
            .groupby("loser_name")
            .tail(1)
        )

        players = {}

        for _, row in winner_df.iterrows():
            players[row["winner_name"]] = {
                "elo": row["elo_winner"],
                "rank": row["winner_rank"],
                "age": row["winner_age"],
                "height": row["winner_ht"],
                "cluster": row["winner_cluster"],
            }

        for _, row in loser_df.iterrows():
            if row["loser_name"] not in players:
                players[row["loser_name"]] = {
                    "elo": row["elo_loser"],
                    "rank": row["loser_rank"],
                    "age": row["loser_age"],
                    "height": row["loser_ht"],
                    "cluster": row["loser_cluster"],
                }

        return players