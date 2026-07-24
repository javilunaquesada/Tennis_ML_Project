import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

CSV_FALLBACK = Path(__file__).parent.parent / "Data" / "processed" / "matches_with_global_elo.csv"


class DatabaseManager:
    """
    Loads match data from PostgreSQL when credentials are available,
    or falls back to the local CSV for environments like Streamlit Cloud
    where no database is configured.
    """

    def __init__(self):
        self.engine = self._create_engine()

    def _create_engine(self):
        """
        Returns a SQLAlchemy engine if all Postgres env vars are set, else None.
        """
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        name = os.getenv("POSTGRES_DB")

        if not all([user, password, host, port, name]):
            return None

        from sqlalchemy import create_engine
        database_url = (
            f"postgresql://{user}:{password}"
            f"@{host}:{int(port)}/{name}?sslmode=require"
        )
        return create_engine(database_url)

    def get_matches(self) -> pd.DataFrame:
        """
        Load all matches from PostgreSQL, or from CSV if no DB is configured.
        """
        if self.engine is not None:
            matches = pd.read_sql("SELECT * FROM matches", self.engine)
        else:
            matches = pd.read_csv(CSV_FALLBACK)

        matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])
        return matches

    def get_latest_players(self) -> dict:
        """
        Build a snapshot of the latest stats for every player.
        """
        matches = self.get_matches()

        winner_df = matches.sort_values("tourney_date").groupby("winner_name").tail(1)
        loser_df = matches.sort_values("tourney_date").groupby("loser_name").tail(1)

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
