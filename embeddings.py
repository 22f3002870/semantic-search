import os
import pickle
import requests
import numpy as np
from config import AIPIPE_BASE_URL, EMBEDDING_MODEL, HEADERS

CACHE_PATH = "cache/embeddings.pkl"

def get_embedding(text):
    response = requests.post(
        f"{AIPIPE_BASE_URL}/embeddings",
        headers=HEADERS,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        }
    )
    return np.array(response.json()["data"][0]["embedding"])

def load_or_create_embeddings(documents):
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    embeddings = []
    for doc in documents:
        emb = get_embedding(doc["content"])
        embeddings.append(emb)

    os.makedirs("cache", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings
