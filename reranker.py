import requests
import json
from config import AIPIPE_BASE_URL, RERANK_MODEL, HEADERS


def rerank(query, candidates):
    """
    Re-rank candidate documents using LLM relevance scoring.
    Returns candidates sorted by normalized score (0–1).
    Never crashes.
    """

    if not candidates:
        return []

    # Build numbered document list
    numbered_docs = ""
    for i, candidate in enumerate(candidates):
        content = candidate.get("content", "")
        numbered_docs += f"\nDocument {i}:\n{content}\n"

    prompt = f"""
You are a relevance scoring assistant.

Query: "{query}"

Below are documents numbered 0 to {len(candidates)-1}.
Rate each document from 0 to 10 based on relevance to the query.

Return results ONLY in this exact JSON format:
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
            timeout=20
        )

        response.raise_for_status()

        result_text = response.json()["choices"][0]["message"]["content"]

    except Exception:
        # If API fails, fallback to original ranking
        for candidate in candidates:
            candidate["score"] = candidate.get("score", 0.0)
        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Try parsing JSON safely
    try:
        scores_json = json.loads(result_text)
        scores = scores_json.get("scores", [])
    except Exception:
        scores = []

    # Assign scores safely
    for i, candidate in enumerate(candidates):
        if i < len(scores):
            try:
                normalized = float(scores[i]) / 10.0
                # Clamp between 0 and 1
                candidate["score"] = max(0.0, min(1.0, normalized))
            except Exception:
                candidate["score"] = 0.0
        else:
            candidate["score"] = 0.0

    return sorted(candidates, key=lambda x: x["score"], reverse=True)
