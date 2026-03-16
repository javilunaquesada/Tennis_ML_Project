# Streamlit App

The app is defined in `app.py` and deployed at:
https://tennismlproject-9v9zbvnmdfk3wcappvru7ej.streamlit.app/

---

## Running Locally

```bash
streamlit run app.py
```

Requires a valid `OPENAI_API_KEY` in a `.env` file at the project root for the LLM explanation feature.

---

## What It Does

1. Loads the trained `MatchOutcomeNN` model and preprocessor from `models/`
2. Builds a player snapshot from `Data/processed/matches_with_global_elo.csv` (latest ELO, rank, age, height, cluster per player)
3. Presents a UI for the user to select two players, a surface, and a tournament level
4. Computes feature differences and runs inference
5. Displays win probabilities for both players with a confidence interpretation
6. Calls the OpenAI API to generate a 2–3 sentence natural language match analysis
7. Shows a visual probability bar and a feature difference summary

---

## UI Components

| Component | Description |
|-----------|-------------|
| Player A / B dropdowns | Sorted list of all players in the dataset |
| Surface selector | Hard, Clay, Grass |
| Tournament level selector | G, M, A, 250, 500 |
| Predict button | Triggers inference |
| Win probability metrics | Side-by-side probability display |
| Confidence label | Balanced / Slight edge / Clear favorite |
| Match Analysis box | LLM-generated explanation |
| Progress bar | Visual probability indicator |
| Caption | ELO diff, rank diff, age diff summary |

---

## LLM Explanation

Handled by `src/llm_explainer.py` via `generate_match_explanation()`.

The function sends a structured prompt to `gpt-4.1-mini` with:
- Predicted probability
- Both players' ELO ratings
- Feature differences (rank, age, height, cluster)
- Surface and tournament level

The model returns a concise analytical explanation suitable for a tennis audience.

---

## Notes

- The model was trained on data up to 2022 and validated on 2023. It is most reliable for 2025 season predictions.
- Performance may degrade for future seasons as player dynamics evolve.
- `@st.cache_resource` is used for the model and preprocessor; `@st.cache_data` for player data.
