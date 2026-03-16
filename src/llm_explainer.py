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
    rank_diff,
    age_diff,
    height_diff,
    cluster_diff
):
    """
    Generate a natural language explanation for the predicted match outcome using an LLM.
    
    """

    prompt = f"""
    You are a tennis analyst.

    A machine learning model has predicted that {player1} has a {probability:.2f}% probability 
    of defeating {player2} on {surface} courts during a {tourney_level} tournament.

    Model statistics:
    {player1} ELO rating: {elo_1}
    {player2} ELO rating: {elo_2}

    Feature differences used by the model (Player1 - Player2):

    Ranking difference: {rank_diff}
    Age difference: {age_diff}
    Height difference: {height_diff}
    Cluster difference: {cluster_diff}



    Explain in 2–3 sentences why the model might favor one player. 
    Base your reasoning primarily on the statistics and feature differences provided above.
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
