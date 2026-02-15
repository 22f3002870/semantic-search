import requests
import json
from config import AIPIPE_BASE_URL, RERANK_MODEL, HEADERS

def rerank(query, candidates):
    if not candidates:
        return candidates

    # Build numbered documents
    numbered_docs = ""
    for i, candidate in enumerate(candidates):
        numbered_docs += f"\nDocument {i}:\n{candidate.get('content', '')}\n"

    prompt = f"""
You are a strict relevance scoring assistant.

Query: "{query}"

Rate each document from 0 to 10 based on relevance to the query.

Return ONLY valid JSON in this exact format:
{{"scores": [score0, score1, score2, ...]}}

Documents:
{numbered_docs}
"""

    try:
        response = requests.post(
            f"{AIPIPE_BASE_URL}/chat/completions",
            headers=HEADERS,
            json={
                "model": RERANK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            },
            timeout=30
        )

        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"]

        # Extract JSON safely
        try:
            scores_json = json.loads(result_text)
            scores = scores_json.get("scores", [])
        except Exception:
            scores = []

    except Exception:
        scores = []

    # Ensure every candidate gets a valid score
    for i, candidate in enumerate(candidates):
        if i < len(scores):
            try:
                normalized = float(scores[i]) / 10.0
                candidate["score"] = max(0.0, min(1.0, normalized))
            except Exception:
                candidate["score"] = 0.0
        else:
            candidate["score"] = 0.0

    return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
