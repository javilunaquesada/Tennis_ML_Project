# Training & Evaluation

Training utilities are in `src/train.py`.

---

## Training Loop

`train_one_epoch(model, loader, optimizer, criterion, device)`

Runs one full pass over the training data:
1. Moves batch to device
2. Computes forward pass
3. Calculates loss (BCEWithLogitsLoss)
4. Backpropagates and updates weights
5. Returns average loss for the epoch

---

## Evaluation

`evaluate(model, loader, device, threshold=0.5)`

Runs inference on a DataLoader and returns:
- `acc` — accuracy at the given threshold
- `auc` — ROC-AUC score
- `all_probs` — raw predicted probabilities
- `all_targets` — ground truth labels

No gradients are computed during evaluation (`torch.no_grad()`).

---

## Data Split Strategy

Splits are strictly temporal to prevent leakage:

| Split | Seasons | Rationale |
|-------|---------|-----------|
| Train | 2018–2022 | Historical data for learning |
| Validation | 2023 | Tune hyperparameters and select best model |
| Test | 2024 | Final evaluation, never seen during training |

ELO ratings are computed sequentially across all seasons before splitting, so each match's ELO only reflects past results.

---

## Datasets

Defined in `src/dataset.py`.

`MatchDataset(X, y)` — basic PyTorch Dataset wrapping feature matrix and labels.

`TennisMatchDatasetWithPlayers(X_numeric, X_categorical, winner_ids, loser_ids, y)` — extended dataset for the embedding model. Returns a 5-tuple per sample.

Helper functions:
- `build_player_id_mapping(matches_df)` — creates consistent `player_name → int` mappings
- `add_player_ids(matches_df, player_to_id)` — adds `winner_id` and `loser_id` columns

---

## Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correctly predicted outcomes at threshold 0.5 |
| ROC-AUC | Area under the ROC curve; primary evaluation metric |

ROC-AUC is preferred because it is threshold-independent and handles the balanced class distribution well.
