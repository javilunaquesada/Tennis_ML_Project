#ELO Implementation — Single Global Rating
import pandas as pd

def compute_elo(matches, k=32, base_rating=1500, rating_col='elo'):
    """
    Compute a global ELO rating for each player and return a dataframe
    with winner and loser ELOs for each match as features.

    matches: pd.DataFrame with columns ["winner_name", "loser_name"]
    k: float, ELO K-factor
    base_rating: initial rating for all players
    rating_col: name of the new ELO feature column (optional)
    """
    # Initialize ratings
    player_ratings = {}
    winner_elo = []
    loser_elo = []

    for _, row in matches.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']

        # Current ratings (default to base)
        rating_winner = player_ratings.get(winner, base_rating)
        rating_loser = player_ratings.get(loser, base_rating)

        # Save current ratings as features
        winner_elo.append(rating_winner)
        loser_elo.append(rating_loser)

        # Expected probabilities
        expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
        expected_loser = 1 - expected_winner

        # Update ratings
        player_ratings[winner] = rating_winner + k * (1 - expected_winner)
        player_ratings[loser] = rating_loser + k * (0 - expected_loser)

    # Add features to dataframe
    matches[f"{rating_col}_winner"] = winner_elo
    matches[f"{rating_col}_loser"] = loser_elo
    matches[f"{rating_col}_diff"] = matches[f"{rating_col}_winner"] - matches[f"{rating_col}_loser"]

    return matches