import torch
from torch.utils.data import Dataset
import pandas as pd

class MatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_player_id_mapping(matches_df):
    """
    Build a mapping from player name to integer ID.
    Ensures consistent indexing for embeddings.
    """
    players = pd.concat([
        matches_df["winner_name"],
        matches_df["loser_name"]
    ]).unique()

    player_to_id = {player: idx for idx, player in enumerate(players)}
    id_to_player = {idx: player for player, idx in player_to_id.items()}

    return player_to_id, id_to_player


def add_player_ids(matches_df, player_to_id):
    """
    Adds winner_id and loser_id columns to matches dataframe.
    """
    matches_df = matches_df.copy()

    matches_df["winner_id"] = matches_df["winner_name"].map(player_to_id)
    matches_df["loser_id"] = matches_df["loser_name"].map(player_to_id)

    return matches_df


class TennisMatchDatasetWithPlayers(Dataset):
    def __init__(
        self,
        X_numeric,
        X_categorical,
        winner_ids,
        loser_ids,
        y
    ):
        self.X_numeric = torch.tensor(X_numeric, dtype=torch.float32)
        self.X_categorical = torch.tensor(X_categorical, dtype=torch.float32)
        self.winner_ids = torch.tensor(winner_ids, dtype=torch.long) # torch.long equivalent to int64, which is required for embedding layers
        self.loser_ids = torch.tensor(loser_ids, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.X_numeric[idx],
            self.X_categorical[idx],
            self.winner_ids[idx],
            self.loser_ids[idx],
            self.y[idx]
        )