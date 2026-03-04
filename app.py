import streamlit as st
import torch
import joblib  # module commonly used for saving/loading scikit-learn models and preprocessors
from pathlib import Path
import sys
import pandas as pd
import json

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
beyond this season, as player dinamics evolve and features vary.
""")

# ---- Project Root Setup ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MatchOutcomeNN

# ---- Load Model and Scaler ----
@st.cache_resource
def load_model_and_preprocessor():
    #MODEL_DIR = PROJECT_ROOT / "models"
    
    # Load processor
    preprocessor = joblib.load(MODEL_DIR / "preprocessor_global_elo.pkl")

    # Load model architecture
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    input_dim = metadata["input_dim"]
    model = MatchOutcomeNN(input_dim)

    state_dict = torch.load(MODEL_DIR / "best_nn_global_elo.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()
st.success("Model and preprocessor loaded successfully.")

# ---- Load Match Data With ELO ----
@st.cache_data
def load_player_data():
    #DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_with_global_elo.csv"
    matches = pd.read_csv(DATA_PATH)

    # Build player snapshot
    players = {}

    for _, row in matches.iterrows():

        # Winner update
        players[row["winner_name"]] = {
            "elo": row["elo_winner"],
            "rank": row["winner_rank"],
            "age": row["winner_age"],
            "height": row["winner_ht"],
            "cluster": row["winner_cluster"]
        }

        # Loser update
        players[row["loser_name"]] = {
            "elo": row["elo_loser"],
            "rank": row["loser_rank"],
            "age": row["loser_age"],
            "height": row["loser_ht"],
            "cluster": row["loser_cluster"]
        }
    
    return players

players_data = load_player_data()
st.success(f"{len(players_data)} players loaded.")
#st.write(list(players_data.keys())[:10])

# Build User Interface for Player Selection
st.header("Match Setup")
player_names = sorted(players_data.keys())

col1, col2 = st.columns(2)

with col1:
    player_a = st.selectbox("Select Player A", player_names)

with col2:
    player_b = st.selectbox("Select Player B", player_names)

# Prevent selecting same player
if player_a == player_b:
    st.warning("Please select two different players.")
    st.stop()

# Surface selection
surface = st.selectbox(
    "Select Surface",
    ["Hard", "Clay", "Grass"]
    )

# Tournament level selection
tourney_level = st.selectbox(
    "Select Tournament Level",
    ["G", "M", "A", "250", "500"]
)

predict_button = st.button("Predict Match Outcome")

# ---- Prediction Logic ----
if predict_button:

    # ---- Extract Player Data ----
    player_a_data =players_data[player_a]
    player_b_data = players_data[player_b]

    # ---- Compute Feature Differences ----
    feature_row = {
        "rank_diff": player_a_data["rank"] - player_b_data["rank"],
        "age_diff": player_a_data["age"] - player_b_data["age"],
        "height_diff": player_a_data["height"] - player_b_data["height"],
        "cluster_diff": player_a_data["cluster"] - player_b_data["cluster"],
        "elo_diff": player_a_data["elo"] - player_b_data["elo"],
        "surface": surface,
        "tourney_level": tourney_level
    }

    # Convert to DataFrame
    feature_df = pd.DataFrame([feature_row])

    # ---- Apply Preprocessor ----
    input_processed = preprocessor.transform(feature_df)

    # Convert to torch tensor
    input_tensor = torch.tensor(input_processed, dtype=torch.float32)

    # ---- Model Prediction ----
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    # ---- Display Results ----
    prob_a = probability
    prob_b = 1 - probability

    st.divider()
    st.subheader("Match Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"{player_a} Win Probability",
            value=f"{prob_a:.2%}"
        )
    with col2:
        st.metric(
            label=f"{player_b} Win Probability",
            value=f"{prob_b:.2%}"
        )

    # Highlight the predicted winner
    if prob_a > prob_b:
        st.success(f"{player_a} is predicted to win!")
    else:
        st.success(f"{player_b} is predicted to win!")

    # Confidence Interpretation
    confidence = abs(prob_a - 0.5)

    if confidence < 0.05:
        st.info("Very balanced matchup")
    elif confidence < 0.15:
        st.info("Slight edge for the predicted winner")
    else:
        st.info("Clear favorite in this matchup based on model predictions")

    # Visual probability bar
    st.progress(prob_a)

    st.caption(f"ELO difference: {feature_row['elo_diff']:.2f} | Rank difference: {feature_row['rank_diff']} | Age difference: {feature_row['age_diff']}")
