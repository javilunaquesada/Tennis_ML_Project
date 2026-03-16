# Project Overview

This project predicts the outcome of professional ATP tennis matches using historical match data (2018–2024). It combines classical machine learning and deep learning, with a focus on feature engineering, temporal validation, and interpretability.

A live Streamlit app lets users select two players, a surface, and a tournament level to get real-time win probability predictions backed by a Neural Network trained with Global ELO ratings.

Live demo: https://tennismlproject-9v9zbvnmdfk3wcappvru7ej.streamlit.app/

---

## Goals

- Explore and understand professional ATP match data
- Cluster players into archetypes based on aggregated statistics
- Predict match outcomes using engineered features
- Compare classical ML vs neural network approaches
- Learn player embeddings to capture latent relational patterns
- Analyze and visualize learned representations

---

## Key Results

| Model                        | Test Accuracy | Test ROC-AUC |
|------------------------------|--------------|--------------|
| Logistic Regression (no ELO) | —            | 0.67         |
| Neural Network (no ELO)      | —            | 0.68         |
| Logistic Regression + ELO    | 0.6466       | 0.7118       |
| Neural Network + ELO         | 0.6489       | **0.7122**   |

The production model is the Neural Network with Global ELO, evaluated on a strictly held-out 2024 test set.

---

## Tech Stack

- Python
- PyTorch
- Scikit-learn
- Pandas / NumPy
- Streamlit
- OpenAI API (GPT-4 mini for match explanations)
- Joblib (model serialization)
