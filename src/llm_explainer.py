import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Load API key from the .env file
load_dotenv()

#Create a client object that interacts with the OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def retrieve_similar_matches(matches, surface, elo_diff, rank_diff):
    
    df = matches.copy()

    df["elo_diff"] = df["elo_winner"] - df["elo_loser"]
    df["rank_diff"] = df["winner_rank"] - df["loser_rank"]

    # Filter by surface
    df = df[df["surface"] == surface]
    
    # Similarity filter
    similar = df[
        (df["elo_diff"].sub(elo_diff).abs() < 100) &  # (df["elo_diff"] - elo_diff).abs() < 100
        (df["rank_diff"].sub(rank_diff).abs() < 30)
    ]

    return similar

def compute_match_stats(similar_matches):
    if len(similar_matches) == 0:
        return None
    
    win_rate = (similar_matches["elo_diff"] > 0).mean() # (creates boolean series).mean()

    return {
        "sample_size": len(similar_matches),
        "win_rate": win_rate
    }

def build_player_embeddings(matches):

    players = {}

    for _, row in matches.iterrows():

        players[row["winner_name"]] = [
            row["elo_winner"],
            row["winner_rank"],
            row["winner_age"],
            row["winner_ht"],
            row["winner_cluster"]
        ]

        players[row["loser_name"]] = [
            row["elo_winner"],
            row["winner_rank"],
            row["winner_age"],
            row["winner_ht"],
            row["winner_cluster"]
        ]

    return players

def find_similar_players(player_name, players_dict, top_k=3):

    if player_name not in players_dict:
        return []

    target = players_dict[player_name]

    similarities = []

    for other, vec in players_dict.items():
        if other == player_name:
            continue
        
        if np.isnan(target).any() or np.isnan(vec).any():
            continue

        sim = cosine_similarity([target], [vec])[0][0]
        similarities.append((other, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return [name for name, _ in similarities[:top_k]]

def generate_match_explanation(
    matches,
    player1,
    player2,
    surface,
    tourney_level,
    probability,
    elo_1,
    elo_2,
    rank_1,
    rank_2,
    age_1,
    age_2,
    height_1,
    height_2,
    cluster_diff
):
    """
    Generate a natural language explanation for the predicted match outcome using an LLM.
    """

    # ---- RAG: MATCH RETRIEVAL ----
    elo_diff = elo_1 - elo_2 if isinstance(elo_1, (int, float)) and isinstance(elo_2, (int, float)) else 0
    rank_diff = (rank_1 - rank_2) if isinstance(rank_1, (int, float)) and isinstance(rank_2, (int, float)) else 0

    similar_matches = retrieve_similar_matches(matches, surface, elo_diff, rank_diff)
    stats = compute_match_stats(similar_matches)

    rag_context = ""
    if stats:
        rag_context = f"""
        Historical matches with similar conditions:
        - Sample size: {stats['sample_size']}
        - In {stats['sample_size']} similar matches, the higher ELO player wins {stats['win_rate']:.2%} of the time
        """
    
    # ---- PLAYER EMBEDDINGS ----
    players_dict = build_player_embeddings(matches)

    similar_p1 = find_similar_players(player1, players_dict)
    similar_p2 = find_similar_players(player2, players_dict)

    embedding_context = f"""
    Players with similar profiles:
    {player1}: {", ".join(similar_p1)}
    {player2}: {", ".join(similar_p2)}
    """

    # ---- FINAL PROMPT ----
    prompt = f"""
    You are a professional tennis analyst.

    A machine learning model predicts that {player1} has a {probability:.2f}% probability 
    of defeating {player2} on {surface} courts in a {tourney_level} tournament.

    Player statistics:
    {player1}: ELO {elo_1} | Rank {rank_1} | Age {age_1} | Height {height_1}
    {player2}: ELO {elo_2} | Rank {rank_2} | Age {age_2} | Height {height_2}

    Cluster difference (Player1 - Player2): {cluster_diff}

    {rag_context}

    {embedding_context}

    Instructions:
    - Explain in 2–3 sentences why the model favors one player.
    - Base your reasoning ONLY on the provided features (ELO, ranking, age, height, cluster) and general playing styles (e.g., surface preferences, playstyle tendencies).
    - DO NOT mention recent form, momentum, injuries, or any unavailable information.
    - If the prediction goes against the higher ELO player, explicitly explain why using the feature differences.
    - Use historical context ({rag_context.strip() != ""}) only as supporting evidence, not as a rule.
    - Be logically consistent with the predicted probability.

    Write a concise, analytical explanation suitable for a tennis audience, and include the probability to contextualize the prediction.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert tennis analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    explanation = response.choices[0].message.content
    return explanation
