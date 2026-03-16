# Feature Engineering

## Match-Level Features

All features are expressed as differences (Player A − Player B) to create a symmetric, direction-aware representation.

| Feature | Description |
|---------|-------------|
| `rank_diff` | ATP ranking difference |
| `age_diff` | Age difference (years) |
| `height_diff` | Height difference (cm) |
| `cluster_diff` | Player cluster ID difference |
| `elo_diff` | Global ELO rating difference |

## Categorical Features (One-Hot Encoded)

| Feature | Values |
|---------|--------|
| `surface` | Hard, Clay, Grass |
| `tourney_level` | G (Grand Slam), M (Masters), A, 250, 500 |

---

## ELO Rating System

Implemented in `src/features.py`.

### Global ELO (`compute_elo`)

A single rating per player, updated sequentially across all matches from 2018 to 2024 before any data split.

Parameters:
- `k = 32` (update factor)
- `base_rating = 1500` (initial rating)

Update formula:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R_A' = R_A + K * (S_A - E_A)
```

Where `S_A = 1` for a win and `0` for a loss.

ELO is computed before splitting to ensure each match only uses past information, preventing leakage.

### Surface-Specific ELO (`compute_surface_elo`)

A separate rating per (player, surface) pair. Tested but did not improve over global ELO, likely due to data sparsity per surface.

---

## Player Clustering

Players are clustered using KMeans (k=5) on aggregated statistics (2018–2023):
- Average ranking
- Win rate
- Average height
- Match participation count

### Cluster Profiles

| Cluster | Avg Rank | Win Rate | Avg Height | Interpretation |
|---------|----------|----------|------------|----------------|
| 0 | 52 | 0.64 | 190 cm | Elite all-rounders |
| 1 | 158 | 0.33 | 183 cm | Lower-tier players |
| 2 | 59 | 0.55 | 200 cm | Big servers / grass specialists |
| 3 | 85 | 0.47 | 183 cm | Solid mid-level competitors |
| 4 | 102 | 0.44 | 190 cm | Inconsistent / specialists |
