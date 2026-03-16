# Data

## Source

ATP match data from [Jeff Sackmann's tennis dataset](https://github.com/JeffSackmann/tennis_atp), covering professional men's tennis matches from 2018 to 2024.

## Raw Files

Located in `Data/raw/`:

| File | Description |
|------|-------------|
| `atp_matches_2018.csv` – `atp_matches_2024.csv` | Match results per season |
| `atp_players.csv` | Player metadata (name, nationality, height, DOB) |

## Temporal Split

| Split      | Seasons     | Purpose                  |
|------------|-------------|--------------------------|
| Train      | 2018–2022   | Model training           |
| Validation | 2023        | Hyperparameter selection |
| Test       | 2024        | Final held-out evaluation|

This temporal split prevents data leakage and reflects real-world forecasting conditions.

## Processed Files

Located in `Data/processed/`:

| File | Description |
|------|-------------|
| `matches_base.csv` | Cleaned match data with basic features |
| `matches_with_global_elo.csv` | Matches augmented with global ELO ratings |
| `matches_with_players_clusters.csv` | Matches with cluster assignments |
| `players_with_clusters.csv` | Player profiles with cluster labels |
| `player_features_final.csv` | Final aggregated player feature matrix |
| `clusters_profile.csv` | Cluster-level summary statistics |
| `feature_names.csv` | Ordered list of feature names used by the model |
| `X_train/val/test.csv` | Feature matrices for each split |
| `y_train/val/test.csv` | Target labels for each split |

## Match Representation

Each match is converted into two symmetric samples to allow the model to learn win probabilities from both perspectives:

- (winner features − loser features) → target = 1
- (loser features − winner features) → target = 0
