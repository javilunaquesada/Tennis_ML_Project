# API Reference

---

## src/features.py

### `compute_elo(matches, k=32, base_rating=1500, rating_col='elo')`

Computes a global ELO rating for each player sequentially over all matches.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `matches` | `pd.DataFrame` | — | Must contain `winner_name` and `loser_name` columns |
| `k` | `float` | `32` | ELO K-factor (controls update magnitude) |
| `base_rating` | `float` | `1500` | Initial rating for all players |
| `rating_col` | `str` | `'elo'` | Prefix for output column names |

Returns the input DataFrame with two new columns: `{rating_col}_winner` and `{rating_col}_loser`.

---

### `compute_surface_elo(matches, k=32, base_rating=1500)`

Computes surface-specific ELO ratings. Each player has a separate rating per surface.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `matches` | `pd.DataFrame` | — | Must contain `winner_name`, `loser_name`, and `surface` columns |
| `k` | `float` | `32` | ELO K-factor |
| `base_rating` | `float` | `1500` | Initial rating per (player, surface) pair |

Returns the input DataFrame with `surface_elo_winner` and `surface_elo_loser` columns.

---

## src/dataset.py

### `MatchDataset(X, y)`

Basic PyTorch `Dataset` wrapping a feature matrix and label vector.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Feature matrix |
| `y` | array-like | Binary labels |

---

### `TennisMatchDatasetWithPlayers(X_numeric, X_categorical, winner_ids, loser_ids, y)`

Extended dataset for the embedding-based model.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_numeric` | array-like | Numeric features |
| `X_categorical` | array-like | One-hot encoded categorical features |
| `winner_ids` | array-like | Integer player IDs for winners |
| `loser_ids` | array-like | Integer player IDs for losers |
| `y` | array-like | Binary labels |

Returns a 5-tuple per `__getitem__` call: `(X_numeric, X_categorical, winner_id, loser_id, y)`.

---

### `build_player_id_mapping(matches_df)`

Builds a bidirectional mapping between player names and integer IDs.

Returns `(player_to_id: dict, id_to_player: dict)`.

---

### `add_player_ids(matches_df, player_to_id)`

Adds `winner_id` and `loser_id` columns to a matches DataFrame using the provided mapping.

Returns a copy of the DataFrame with the new columns.

---

## src/models.py

### `MatchOutcomeNN(input_dim)`

Feedforward neural network for binary match outcome prediction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dim` | `int` | Number of input features |

`forward(x)` — takes a float tensor of shape `(batch, input_dim)`, returns logits of shape `(batch,)`.

---

### `MatchOutcomeNNWithEmbeddings(num_numeric_features, num_categorical_features, num_players, embedding_dim=8, hidden_dim=32, dropout=0.3)`

Neural network with learnable player embeddings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_numeric_features` | `int` | — | Number of numeric input features |
| `num_categorical_features` | `int` | — | Number of one-hot encoded features |
| `num_players` | `int` | — | Total number of unique players (embedding vocabulary size) |
| `embedding_dim` | `int` | `8` | Dimensionality of player embeddings |
| `hidden_dim` | `int` | `32` | Hidden layer size |
| `dropout` | `float` | `0.3` | Dropout rate for first hidden layer |

`forward(x_num, x_cat, winner_id, loser_id)` — returns logits of shape `(batch,)`.

---

## src/train.py

### `train_one_epoch(model, loader, optimizer, criterion, device)`

Runs one training epoch.

Returns average loss over all batches (`float`).

---

### `evaluate(model, loader, device, threshold=0.5)`

Evaluates a model on a DataLoader.

Returns `(accuracy, roc_auc, all_probs, all_targets)`.

---

## src/llm_explainer.py

### `generate_match_explanation(matches, player1, player2, surface, tourney_level, probability, elo_1, elo_2, rank_diff, age_diff, height_diff, cluster_diff)`

Generates a natural language match analysis using OpenAI GPT-4 mini.

| Parameter | Type | Description |
|-----------|------|-------------|
| `player1` / `player2` | `str` | Player names |
| `surface` | `str` | Match surface |
| `tourney_level` | `str` | Tournament level code |
| `probability` | `float` | Predicted win probability for player1 (0–100) |
| `elo_1` / `elo_2` | `float` | ELO ratings |
| `rank_diff` | `float` | Ranking difference |
| `age_diff` | `float` | Age difference |
| `height_diff` | `float` | Height difference |
| `cluster_diff` | `float` | Cluster ID difference |

Returns a string with the generated explanation.

Requires `OPENAI_API_KEY` set in the environment (loaded from `.env`).
