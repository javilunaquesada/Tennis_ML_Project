import os
from openai import OpenAI
from dotenv import load_dotenv

#Load API key from the .env file
load_dotenv()

#Create a client object that interacts with the OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_match_explanation(
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

    prompt = f"""
    You are a tennis analyst.

    A machine learning model has predicted that {player1} has a {probability:.2f}% probability 
    of defeating {player2} on {surface} courts during a {tourney_level} tournament.

    Player statistics:
    {player1}: ELO {elo_1:.1f} | Rank {rank_1} | Age {age_1:.1f} | Height {height_1:.0f} cm
    {player2}: ELO {elo_2:.1f} | Rank {rank_2} | Age {age_2:.1f} | Height {height_2:.0f} cm

    Additional model input:
    Cluster difference (Player1 - Player2): {cluster_diff}

    Explain in 2–3 sentences why the model might favor one player.
    Base your reasoning primarily on the statistics provided above.
    Mention the predicted probability to contextualize the analysis and highlight the most
    important factors influencing the prediction.

    Be concise, analytical, and clear for a tennis audience.
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
