import torch
import torch.nn as nn

class MatchOutcomeNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

class MatchOutcomeNNWithEmbeddings(nn.Module):
    def __init__(
        self, 
        num_numeric_features, 
        num_categorical_features, 
        num_players, 
        embedding_dim=8, 
        hidden_dim=32, 
        dropout=0.3
    ):
        super().__init__()

        # Player embeddings
        self.player_embedding = nn.Embedding(
            num_embeddings=num_players, 
            embedding_dim=embedding_dim
        )

        # Total input size:
        # numeric + categorical + embedding_diff
        input_dim = (
            num_numeric_features + 
            num_categorical_features + 
            embedding_dim
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        x_num,
        x_cat,
        winner_id,
        loser_id
    ):
        # Look up embeddings
        winner_emb = self.player_embedding(winner_id)
        loser_emb = self.player_embedding(loser_id)

        # Difference of embeddings
        emb_diff = winner_emb - loser_emb

        # Concatenate all inputs
        x = torch.cat(
            [x_num, x_cat, emb_diff],
            dim=1
        )

        logits = self.network(x)
        return logits.squeeze(1)