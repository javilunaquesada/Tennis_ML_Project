# Notebooks

The `Notebooks/` folder contains a progressive sequence of experiments, from raw data exploration to the final production model.

---

## 01 — Data Understanding

`01_data_understanding.ipynb`

Exploratory analysis of ATP match data (2018–2024):
- Match distributions by surface, tournament level, and year
- Player longevity and participation counts
- Ranking dynamics and outcome patterns
- Surface effects on match statistics

---

## 02 — Feature Engineering

`02_feature_engineering.ipynb`

Constructs the base feature set:
- Aggregates player-level statistics (win rate, avg ranking, height, age)
- Creates match-level difference features (rank diff, age diff, height diff)
- One-hot encodes surface and tournament level
- Generates symmetric match samples (winner vs loser → 1, loser vs winner → 0)

---

## 03 — Player Clustering

`03_player_clustering.ipynb`

Unsupervised player profiling:
- Applies KMeans (k=5) on aggregated player statistics
- Identifies 5 player archetypes (see [Feature Engineering](features.md) for cluster profiles)
- Visualizes clusters via PCA projection
- Adds cluster membership as a match feature

---

## 04 — Match Prediction (Scikit-learn)

`04_match_prediction_sklearn.ipynb`

Classical ML baseline:
- Trains Logistic Regression on engineered features
- Evaluates on validation and test sets
- Baseline ROC-AUC ≈ 0.67

---

## 05 — Match Prediction (PyTorch)

`05_match_prediction_pytorch.ipynb`

Neural network baseline:
- Builds and trains `MatchOutcomeNN`
- Compares against Logistic Regression
- Test ROC-AUC ≈ 0.68

---

## 06 — Player Embeddings Analysis

`06_player_embeddings_analysis.ipynb`

Representation learning:
- Trains `MatchOutcomeNNWithEmbeddings`
- Visualizes learned embeddings via PCA
- Compares embedding space against KMeans cluster labels
- Analyzes player similarity (nearest neighbors in embedding space)

Key finding: embeddings form a continuous cloud rather than discrete groups, reflecting the gradient nature of tennis skill.

---

## 07 — Match Prediction with ELO

`07_match_prediction_with_elo.ipynb`

Production model development:
- Implements global ELO rating system (`compute_elo`)
- Computes ELO sequentially across 2018–2024 before splitting
- Trains Logistic Regression and `MatchOutcomeNN` with ELO feature
- ROC-AUC improves from ~0.68 to ~0.71 (+0.03)
- Saves the best model, preprocessor, and metadata to `models/`

---

## 08 — Surface-Specific ELO Experiment

`08_match_prediction_surface_elo.ipynb`

Ablation study:
- Implements `compute_surface_elo` (separate ratings per player per surface)
- Trains and evaluates models with surface-specific ELO
- Result: no improvement over global ELO (ROC-AUC 0.711 vs 0.712)
- Conclusion: global strength dynamics are sufficient for this dataset size
