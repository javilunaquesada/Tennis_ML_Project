import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_DIR = Path(__file__).parent.parent / "models"
INDEX_PATH = CACHE_DIR / "rag_index.pkl"

EMBED_MODEL = "text-embedding-3-small"


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _embed_texts(texts: list) -> np.ndarray:
    vectors = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return np.array(vectors, dtype=np.float32)


def _embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array([response.data[0].embedding], dtype=np.float32)


# ── Document builders ──────────────────────────────────────────────────────────

def _build_match_doc(row: pd.Series) -> str:
    return (
        f"{row['winner_name']} defeated {row['loser_name']} "
        f"on {row.get('surface', 'unknown')} surface "
        f"in a {row.get('tourney_level', 'unknown')} tournament. "
        f"Winner ELO: {row.get('elo_winner', 0):.0f}, "
        f"Loser ELO: {row.get('elo_loser', 0):.0f}. "
        f"Winner rank: {row.get('winner_rank', 'N/A')}, "
        f"Loser rank: {row.get('loser_rank', 'N/A')}."
    )


def _build_player_doc(name: str, stats: dict) -> str:
    return (
        f"Player {name}: "
        f"ELO {stats.get('elo', 0):.0f}, "
        f"rank {stats.get('rank', 'N/A')}, "
        f"age {stats.get('age', 'N/A')}, "
        f"height {stats.get('height', 'N/A')} cm, "
        f"cluster {stats.get('cluster', 'N/A')}."
    )


# ── Pure numpy cosine index ────────────────────────────────────────────────────

class _NumpyIndex:
    def __init__(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self._normed = vectors / norms

    def search(self, query: np.ndarray, k: int):
        q = query / (np.linalg.norm(query) + 1e-9)
        scores = (self._normed @ q.T).squeeze()
        top_k = min(k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return scores[indices], indices


# ── Main pipeline class ────────────────────────────────────────────────────────

class TennisRAGPipeline:
    """
    Production RAG pipeline. Builds semantic indexes over match history
    and player profiles using OpenAI embeddings, persisted to disk.
    """

    def __init__(self):
        self.match_docs = []
        self.match_meta = []
        self.match_index = None
        self.player_docs = []
        self.player_names = []
        self.player_index = None

    def build(self, matches: pd.DataFrame, players: dict, force_rebuild: bool = False):
        if not force_rebuild and INDEX_PATH.exists():
            self._load()
            return

        self.match_docs = [_build_match_doc(row) for _, row in matches.iterrows()]
        self.match_meta = matches[
            ["winner_name", "loser_name", "surface", "tourney_level",
             "elo_winner", "elo_loser", "winner_rank", "loser_rank"]
        ].to_dict(orient="records")

        match_vectors = _embed_texts(self.match_docs)
        self.match_index = _NumpyIndex(match_vectors)

        self.player_names = list(players.keys())
        self.player_docs = [_build_player_doc(n, players[n]) for n in self.player_names]
        player_vectors = _embed_texts(self.player_docs)
        self.player_index = _NumpyIndex(player_vectors)

        self._save(match_vectors, player_vectors)

    def _save(self, match_vectors, player_vectors):
        payload = {
            "match_docs": self.match_docs,
            "match_meta": self.match_meta,
            "match_vectors": match_vectors,
            "player_docs": self.player_docs,
            "player_names": self.player_names,
            "player_vectors": player_vectors,
        }
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(payload, f)

    def _load(self):
        with open(INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
        self.match_docs = payload["match_docs"]
        self.match_meta = payload["match_meta"]
        self.match_index = _NumpyIndex(payload["match_vectors"])
        self.player_docs = payload["player_docs"]
        self.player_names = payload["player_names"]
        self.player_index = _NumpyIndex(payload["player_vectors"])

    def retrieve_similar_matches(self, player1, player2, surface, tourney_level, top_k=5):
        if self.match_index is None:
            return []
        query = f"Match between {player1} and {player2} on {surface} in a {tourney_level} tournament."
        q_vec = _embed_query(query)
        scores, indices = self.match_index.search(q_vec, k=top_k * 3)

        results = []
        for score, idx in zip(scores, indices):
            meta = self.match_meta[idx]
            if meta.get("surface") == surface:
                results.append({**meta, "similarity": float(score)})
            if len(results) >= top_k:
                break

        if len(results) < top_k:
            for score, idx in zip(scores, indices):
                meta = self.match_meta[idx]
                entry = {**meta, "similarity": float(score)}
                if entry not in results:
                    results.append(entry)
                if len(results) >= top_k:
                    break

        return results

    def retrieve_similar_players(self, player_name, top_k=3):
        if self.player_index is None or player_name not in self.player_names:
            return []
        idx_self = self.player_names.index(player_name)
        q_vec = _embed_query(self.player_docs[idx_self])
        _, indices = self.player_index.search(q_vec, k=top_k + 1)
        return [self.player_names[i] for i in indices if self.player_names[i] != player_name][:top_k]

    def build_rag_context(self, player1, player2, surface, tourney_level) -> str:
        similar_matches = self.retrieve_similar_matches(player1, player2, surface, tourney_level)
        similar_p1 = self.retrieve_similar_players(player1)
        similar_p2 = self.retrieve_similar_players(player2)

        if similar_matches:
            higher_elo_wins = sum(1 for m in similar_matches if m.get("elo_winner", 0) > m.get("elo_loser", 0))
            win_rate = higher_elo_wins / len(similar_matches)
            match_context = (
                f"Semantically similar historical matches ({len(similar_matches)} found):\n"
                f"- Higher-ELO player won {win_rate:.0%} of those matches.\n"
            )
            for m in similar_matches[:3]:
                match_context += (
                    f"  • {m['winner_name']} def. {m['loser_name']} "
                    f"on {m.get('surface','?')} "
                    f"(ELO diff: {m.get('elo_winner',0) - m.get('elo_loser',0):.0f}, "
                    f"similarity: {m['similarity']:.2f})\n"
                )
        else:
            match_context = "No semantically similar historical matches found.\n"

        p1_str = ", ".join(similar_p1) if similar_p1 else "none found"
        p2_str = ", ".join(similar_p2) if similar_p2 else "none found"
        player_context = (
            f"Players with similar profiles (semantic search):\n"
            f"- {player1}: {p1_str}\n"
            f"- {player2}: {p2_str}\n"
        )

        return match_context + "\n" + player_context
