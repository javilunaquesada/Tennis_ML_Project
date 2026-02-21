import pandas as pd

#ELO Implementation — Single Global Rating

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

    return matches


#ELO Implementation — Surface-Specific Ratings

def compute_surface_elo(matches, k=32, base_rating=1500):
    """
    Compute surface-specific ELO ratings.
    Each player has a separate rating per surface.
    """
    # Initialize ratings
    player_ratings = {}
    winner_elo = []
    loser_elo = []

    for _, row in matches.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row["surface"]

        key_w = (winner, surface)
        key_l = (loser, surface)

        rating_winner = player_ratings.get(key_w, base_rating) # if key_w exists, get rating, else use base_rating
        rating_loser = player_ratings.get(key_l, base_rating)  # if key_l exists, get rating, else use base_rating

        winner_elo.append(rating_winner)
        loser_elo.append(rating_loser)

        # Expected probabilities
        expected_winner = 1 / (1 + 10 ** ((rating_loser - rating_winner) / 400))
        expected_loser = 1 - expected_winner

        # Update ratings
        player_ratings[key_w] = rating_winner + k * (1 - expected_winner)
        player_ratings[key_l] = rating_loser + k * (0 - expected_loser)

    matches["surface_elo_winner"] = winner_elo
    matches["surface_elo_loser"] = loser_elo
    print(player_ratings)
    print(winner_elo)
    print(loser_elo)
    return matches