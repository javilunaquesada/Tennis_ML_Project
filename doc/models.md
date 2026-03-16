# Models

All model architectures are defined in `src/models.py`.

---

## MatchOutcomeNN

Simple feedforward neural network used for match outcome prediction.

```
Input (input_dim)
  → Linear(input_dim, 32) → ReLU → Dropout(0.3)
  → Linear(32, 16)        → ReLU → Dropout(0.2)
  → Linear(16, 1)
  → Sigmoid (applied at inference)
```

Output is a single logit; sigmoid is applied externally to get win probability.

This is the production model, trained with global ELO features.

---

## MatchOutcomeNNWithEmbeddings

Extended architecture that learns a latent representation for each player.

```
winner_id → Embedding(num_players, embedding_dim)  ─┐
loser_id  → Embedding(num_players, embedding_dim)  ─┤ → diff
                                                     │
x_numeric ──────────────────────────────────────────┤
x_categorical ──────────────────────────────────────┘
  → Concat → Linear(input_dim, 32) → ReLU → Dropout(0.3)
           → Linear(32, 16)        → ReLU → Dropout(0.15)
           → Linear(16, 1)
```

Parameters:
- `embedding_dim = 8`
- `hidden_dim = 32`
- `dropout = 0.3`

The embedding difference (`winner_emb − loser_emb`) is concatenated with numeric and categorical features before the MLP layers.

This model enables player similarity analysis and embedding visualization, but achieved slightly lower predictive performance than the simpler model due to higher complexity relative to dataset size.

---

## Saved Model Artifacts

Located in `models/`:

| File | Description |
|------|-------------|
| `best_nn_global_elo.pth` | Trained weights for `MatchOutcomeNN` with ELO |
| `preprocessor_global_elo.pkl` | Fitted scikit-learn preprocessor (scaler + encoder) |
| `model_metadata.json` | Input dimension and other metadata needed to reconstruct the model |
