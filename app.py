import streamlit as st
import torch
import joblib
from pathlib import Path
import sys
import pandas as pd
import json
import numpy as np
from src.llm_explainer import generate_match_explanation

# ---- Page Configuration ----

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "Data" / "processed" / "matches_with_global_elo.csv"

st.set_page_config(page_title="Tennis Match Predictor", layout="centered")

st.title("🎾 Tennis Match Outcome Predictor")

st.write("""
This app predicts the probability of one player defeating another
using a Neural Network trained with Global ELO ratings.
""")

st.info("""
This model was trained with data previous to 2024, and evaluated on matches from 2024, 
so it is most reliable for predictions regarding the 2025 season. Performance may decrease for matches
beyond this season, as player dynamics evolve and features vary.
""")

# ---- Project Root Setup ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MatchOutcomeNN

# ---- Load Model and Preprocessor ----
@st.cache_resource
def load_model_and_preprocessor():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor_global_elo.pkl")

    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    model = MatchOutcomeNN(metadata["input_dim"])

    state_dict = torch.load(MODEL_DIR / "best_nn_global_elo.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()
st.success("Model and preprocessor loaded successfully.")

# ---- Load Player Data ----
@st.cache_data
def load_player_data():
    matches = pd.read_csv(DATA_PATH)
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])

    winner_df = matches.sort_values("tourney_date").groupby("winner_name").tail(1)
    loser_df = matches.sort_values("tourney_date").groupby("loser_name").tail(1)

    players = {}

    for _, row in winner_df.iterrows():
        players[row["winner_name"]] = {
            "elo": row["elo_winner"],
            "rank": row["winner_rank"],
            "age": row["winner_age"],
            "height": row["winner_ht"],
            "cluster": row["winner_cluster"]
        }

    for _, row in loser_df.iterrows():
        if row["loser_name"] not in players:
            players[row["loser_name"]] = {
                "elo": row["elo_loser"],
                "rank": row["loser_rank"],
                "age": row["loser_age"],
                "height": row["loser_ht"],
                "cluster": row["loser_cluster"]
            }

    return players, matches

players_data, matches = load_player_data()
st.success(f"{len(players_data)} players loaded.")

# ---- UI ----
st.header("Match Setup")
player_names = sorted(players_data.keys())

col1, col2 = st.columns(2)
with col1:
    player_a = st.selectbox("Select Player A", player_names)
with col2:
    player_b = st.selectbox("Select Player B", player_names)

if player_a == player_b:
    st.warning("Please select two different players.")
    st.stop()

surface = st.selectbox("Select Surface", ["Hard", "Clay", "Grass"])
tourney_level = st.selectbox("Select Tournament Level", ["G", "M", "A", "250", "500"])

predict_button = st.button("Predict Match Outcome")

# ---- Prediction ----
if predict_button:

    player_a_data = players_data[player_a]
    player_b_data = players_data[player_b]

    # ✅ Explicit numeric values
    rank_a, rank_b = float(player_a_data["rank"]), float(player_b_data["rank"])
    age_a, age_b = float(player_a_data["age"]), float(player_b_data["age"])
    height_a, height_b = float(player_a_data["height"]), float(player_b_data["height"])
    elo_a, elo_b = float(player_a_data["elo"]), float(player_b_data["elo"])
    cluster_a, cluster_b = float(player_a_data["cluster"]), float(player_b_data["cluster"])

    # ✅ Explicit diffs (single source of truth)
    rank_diff = rank_a - rank_b
    age_diff = age_a - age_b
    height_diff = height_a - height_b
    cluster_diff = cluster_a - cluster_b
    elo_diff = elo_a - elo_b

    # Debug (remove later)
    st.write("DEBUG:", player_a, rank_a, "|", player_b, rank_b, "| diff:", rank_diff)

    # ---- Model input ----
    feature_df = pd.DataFrame([{
        "rank_diff": rank_diff,
        "age_diff": age_diff,
        "height_diff": height_diff,
        "cluster_diff": cluster_diff,
        "elo_diff": elo_diff,
        "surface": surface,
        "tourney_level": tourney_level
    }])

    input_processed = preprocessor.transform(feature_df)
    input_tensor = torch.tensor(input_processed, dtype=torch.float32)

    with torch.no_grad():
        probability = torch.sigmoid(model(input_tensor)).item()

    probability = float(np.clip(probability, 0, 1))
    prob_a, prob_b = probability, 1 - probability

    # ---- UI ----
    st.divider()
    st.subheader("Match Prediction")

    col1, col2 = st.columns(2)
    col1.metric(f"{player_a} Win Probability", f"{prob_a:.2%}")
    col2.metric(f"{player_b} Win Probability", f"{prob_b:.2%}")

    st.success(f"{player_a if prob_a > prob_b else player_b} is predicted to win!")

    confidence = abs(prob_a - 0.5)
    if confidence < 0.05:
        st.info("Very balanced matchup")
    elif confidence < 0.15:
        st.info("Slight edge")
    else:
        st.info("Clear favorite")

    # ---- LLM ----
    def safe_val(v, fallback="N/A"):
        return v if pd.notna(v) else fallback

    explanation = generate_match_explanation(
        matches=matches,
        player1=player_a,
        player2=player_b,
        surface=surface,
        tourney_level=tourney_level,
        probability=prob_a * 100,
        elo_1=elo_a,
        elo_2=elo_b,
        rank_1=rank_a,
        rank_2=rank_b,
        rank_diff=rank_diff,  # ✅ KEY FIX
        age_1=age_a,
        age_2=age_b,
        height_1=height_a,
        height_2=height_b,
        cluster_diff=cluster_diff
    )

    st.subheader("Match Analysis")
    st.info(explanation)

    if not np.isnan(prob_a):
        st.progress(prob_a)
        st.write(f"{prob_a*100:.1f}%")

    # ✅ ALWAYS CORRECT NOW
    st.caption(
        f"ELO diff: {elo_diff:.2f} | Rank diff: {rank_diff:.2f} | Age diff: {age_diff:.2f}"
    )