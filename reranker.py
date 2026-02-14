import requests
from config import AIPIPE_BASE_URL, RERANK_MODEL, HEADERS

def rerank(query, candidates):
    numbered_docs = ""
    for i, candidate in enumerate(candidates):
        numbered_docs += f"\nDocument {i}:\n{candidate['content']}\n"

    prompt = f"""
You are a relevance scoring assistant.

Query: "{query}"

Below are documents numbered 0 to {len(candidates)-1}.
Rate each document from 0 to 10 based on relevance to the query.

Return results in this exact JSON format:
{{"scores": [score0, score1, score2, ...]}}

Documents:
{numbered_docs}
"""

    response = requests.post(
        f"{AIPIPE_BASE_URL}/chat/completions",
        headers=HEADERS,
        json={
            "model": RERANK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )

    result_text = response.json()["choices"][0]["message"]["content"]

    # Extract JSON safely
    import json
    scores_json = json.loads(result_text)
    scores = scores_json["scores"]

    for i, candidate in enumerate(candidates):
        candidate["score"] = float(scores[i]) / 10.0

    return sorted(candidates, key=lambda x: x["score"], reverse=True)
