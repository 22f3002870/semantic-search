import json
import numpy as np
import time
from embeddings import load_or_create_embeddings, get_embedding
from reranker import rerank

with open("data/reviews.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_embeddings = load_or_create_embeddings(documents)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def normalize_scores(results):
    scores = [r["score"] for r in results]
    max_s = max(scores)
    min_s = min(scores)

    for r in results:
        if max_s == min_s:
            r["score"] = 1.0
        else:
            r["score"] = (r["score"] - min_s) / (max_s - min_s)

    return results

def semantic_search(data):
    start_time = time.time()

    query = data.get("query", "")
    k = data.get("k", 10)
   
    rerank_flag = data.get("rerank", False)

    rerankK = data.get("rerankK", 6)

    if not query:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    query_embedding = get_embedding(query)

    similarities = []
    for idx, emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, emb)
        similarities.append((idx, score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_k = similarities[:k]

    results = []
    for idx, score in top_k:
        results.append({
            "id": documents[idx]["id"],
            "score": score,
            "content": documents[idx]["content"],
            "metadata": {"source": documents[idx].get("source", "reviews.json")}
        })

    results = normalize_scores(results)

    reranked = False

    if rerank_flag and results:
        reranked = True
        results = rerank(query, results)
        results = results[:rerankK]

    latency = int((time.time() - start_time) * 1000)
    # Ensure every result has valid numeric score between 0 and 1
    for r in results:
        try:
            score = float(r.get("score", 0.0))
            if score != score:  # check NaN
                score = 0.0
            r["score"] = max(0.0, min(1.0, score))
        except:
            r["score"] = 0.0

    return {
        "results": results,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
