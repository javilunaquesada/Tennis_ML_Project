import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from src.rag_pipeline import TennisRAGPipeline

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Module-level pipeline instance — built once, reused across calls
_pipeline = None


def get_pipeline(matches: pd.DataFrame, players: dict) -> TennisRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TennisRAGPipeline()
        _pipeline.build(matches, players)
    return _pipeline


def generate_match_explanation(
    matches: pd.DataFrame,
    players: dict,
    player1: str,
    player2: str,
    surface: str,
    tourney_level: str,
    probability: float,
    elo_1,
    elo_2,
    rank_1,
    rank_2,
    age_1,
    age_2,
    height_1,
    height_2,
    cluster_diff,
) -> str:
    """
    Generate a natural language explanation for the predicted match outcome.
    Uses the RAG pipeline for semantic retrieval of relevant historical context.
    """
    pipeline = get_pipeline(matches, players)
    rag_context = pipeline.build_rag_context(player1, player2, surface, tourney_level)

    elo_diff    = elo_1 - elo_2       if isinstance(elo_1,    (int, float)) and isinstance(elo_2,    (int, float)) else 0
    rank_diff   = rank_1 - rank_2     if isinstance(rank_1,   (int, float)) and isinstance(rank_2,   (int, float)) else 0
    age_diff    = age_1 - age_2       if isinstance(age_1,    (int, float)) and isinstance(age_2,    (int, float)) else "N/A"
    height_diff = height_1 - height_2 if isinstance(height_1, (int, float)) and isinstance(height_2, (int, float)) else "N/A"

    prompt = f"""
You are a tennis analyst.

A machine learning model predicts that {player1} has a {probability:.2f}% probability
of defeating {player2} on {surface} courts in a {tourney_level} tournament.

Player statistics:
{player1}: ELO {elo_1} | Rank {rank_1} | Age {age_1} | Height {height_1}
{player2}: ELO {elo_2} | Rank {rank_2} | Age {age_2} | Height {height_2}

Feature differences (Player1 - Player2):
- Rank difference: {rank_diff}
- Age difference: {age_diff}
- Height difference: {height_diff}
- Cluster difference: {cluster_diff}
- ELO difference: {elo_diff}

IMPORTANT:
- A NEGATIVE rank difference means Player 1 has a BETTER ranking.
- A POSITIVE rank difference means Player 1 has a WORSE ranking.
- A POSITIVE ELO difference means Player 1 is stronger.
- A NEGATIVE ELO difference means Player 2 is stronger.

Retrieved context (RAG):
{rag_context}

Explain in 2–3 sentences why the model favors one player.

Only use the features provided above.
Interpret the sign of each feature correctly based on the definitions.
Do NOT assume that negative values are bad or positive values are good without context.
Do NOT mention momentum, injuries, or recent form.
Do NOT use cluster information for this analysis.

Be precise, consistent with the data, and avoid contradictions.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert tennis analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content
