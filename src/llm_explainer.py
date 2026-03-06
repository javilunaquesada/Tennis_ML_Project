import os
from openai import OpenAI
from dotenv import load_dotenv

#Load API key from the .env file
load_dotenv()

#Create a client object that interacts with the OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 


def generate_match_explanation(player1, player2, surface, tourney_level, probability):
    """
    Generate a natural language explanation for the predicted match outcome using an LLM.
    
    """

    prompt = f"""
    You are a tennis analyst. 
    A machine learning model has predicted that {player1} has a {probability:.2f}% probability 
    of defeating {player2} on {surface} courts during a {tourney_level} tournament.

    Explain in 2-3 sentences what factors might justify this prediction, considering player styles,
    surface preferences, and 2024 performance trends. Focus on the most relevant aspects that could 
    influence the outcome.
    Be concise and informative, providing insights that a tennis fan would find interesting. Include
    the probability numbers in your explanation to contextualize the analysis.
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
