# 🎾 Tennis Match Outcome Prediction & Player Embeddings

## 📌 Project Overview

This project explores **professional tennis match data** to analyze player performance, identify player archetypes, and predict match outcomes using both **classical machine learning** and **deep learning models**.

Using historical ATP match data (2018–2024), we progressively build:
- Player-level statistical profiles
- Unsupervised player clusters
- Match-level predictive models
- Neural networks with **learned player embeddings**

The project emphasizes **rigorous data splitting**, **model comparison**, and **interpretability**, following best practices used in real-world data science projects.

---

## 🎯 Objectives

- Perform exploratory analysis of professional tennis match data  
- Cluster players based on aggregated performance statistics  
- Predict match outcomes using engineered features  
- Compare classical ML models vs neural networks  
- Learn **player embeddings** to capture latent relational patterns  
- Analyze and visualize learned representations  

---

## 🗂️ Project Structure

Tennis_ML_Project/
│
├── Data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_player_clustering.ipynb
│   ├── 04_match_prediction_sklearn.ipynb
│   ├── 05_match_prediction_pytorch.ipynb
│   └── 06_match_prediction_pytorch_bonus.ipynb
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│
└── README.md


---

## 📊 Data Source

- **Jeff Sackmann’s ATP Tennis Dataset**
- Covers professional men’s tennis matches
- Includes match results, player rankings, surfaces, tournament levels, and basic stats

**Time range used:**
- **2018–2023** → player statistics & clustering  
- **2018–2022** → training set  
- **2023** → validation set  
- **2024** → test set (strictly held out)

This temporal split avoids data leakage and reflects real-world forecasting conditions.

---

## 🔍 Exploratory Data Analysis

Key insights:
- Match outcomes are strongly influenced by ranking differences  
- Player statistics vary significantly by surface  
- Tennis performance lies on **continuous gradients**, not discrete categories

EDA notebooks focus on:
- Match distributions  
- Player longevity and participation  
- Surface effects  
- Ranking dynamics

---

## 🧩 Player Clustering

Players are clustered using **KMeans** based on aggregated statistics (2018–2023), such as:
- Average ranking  
- Age  
- Height  
- Match participation

Clustering reveals broad archetypes (e.g. elite players, regular tour players, fringe players), but with **significant overlap**, reflecting the continuous nature of tennis skill.

---

## ⚙️ Match Outcome Prediction

### Features
- Rank difference  
- Age difference  
- Height difference  
- Player cluster difference  
- Surface (One-Hot encoded)  
- Tournament level (One-Hot encoded)

Each match is transformed into **two samples**:
- Winner vs Loser → target = 1  
- Loser vs Winner → target = 0

This framing allows symmetric learning of win probabilities.

---

## 📈 Models & Results

### Logistic Regression (baseline)
- Validation ROC-AUC ≈ **0.67**
- Test ROC-AUC (2024) ≈ **0.67**

### Neural Network (PyTorch)
- Fully connected network  
- Improved flexibility over linear models  
- Test ROC-AUC ≈ **0.68**

### Neural Network with Player Embeddings
- Introduces learnable embeddings for each player  
- Captures relational information between players  
- Slightly lower predictive performance due to higher model complexity and limited data  
- Enables deeper **representation analysis**

---

## 🧠 Player Embeddings Analysis

Learned player embeddings are:
- Visualized using PCA  
- Compared against KMeans clusters  
- Used to compute player similarity

Key findings:
- Embedding space is **continuous**, not sharply clustered  
- Embeddings capture relational patterns not explained by static statistics  
- Elite players tend to cluster near each other, but overlap remains high

Example:
- Rafael Nadal’s nearest neighbors include Novak Djokovic and Juan Martín del Potro, reflecting shared competitive contexts.

## PCA Projection of Player Embeddings

![Player Embeddings PCA](images/player_embeddings_pca.png)

The embedding space forms a continuous cloud rather than sharply separated groups.

This indicates that tennis skill and player characteristics exist on gradients rather than discrete categories.

---

## Embeddings vs KMeans Clusters

![Embeddings Colored by Clusters](images/embeddings_vs_clusters.png)

When coloring embeddings by cluster label:

- There is noticeable overlap.
- Embeddings do not perfectly replicate clustering.

This suggests that embeddings capture relational match dynamics rather than static player attributes.

Clustering describes *who players are statistically*.  
Embeddings describe *how players interact competitively*.

---

## 🧪 Evaluation Metrics

- Accuracy  
- ROC-AUC  
- Confusion matrices  
- Validation-driven model selection

The **2024 season** is kept fully unseen until final evaluation.

---

## 🧠 Key Takeaways

- Simple features already carry strong predictive signal in tennis  
- Neural networks offer marginal gains over linear models  
- Player embeddings provide **interpretability and insight**, even when predictive gains are limited  
- Model complexity must be balanced carefully with dataset size  
- Temporal validation is essential in sports analytics

---

## 🚀 Future Work

- Surface-specific player embeddings  
- Sequence-based models (match history as time series)  
- Bayesian or Elo-style hybrid models  
- Women’s tennis (WTA) extension  
- Live betting probability calibration

---

## 🛠️ Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- PyTorch  
- Matplotlib / Seaborn

---

## 👤 Author

**Javier Luna Quesada**  
Data Science & Machine Learning Enthusiast  
🎾 Sports analytics | 🤖 Applied ML | 📊 Data-driven insights