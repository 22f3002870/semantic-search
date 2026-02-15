import requests
import json
import re
from config import AIPIPE_BASE_URL, RERANK_MODEL, HEADERS

def rerank(query, candidates):
    if not candidates:
        return []

    # Build prompt
    numbered_docs = ""
    for i, candidate in enumerate(candidates):
        numbered_docs += f"\nDocument {i}:\n{candidate['content']}\n"

    prompt = f"""
You are a strict relevance scoring assistant.

Query: "{query}"

Rate each document from 0 to 10.

Return ONLY valid JSON:
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

        content = response.json()["choices"][0]["message"]["content"]

        # Extract JSON safely using regex
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            scores_json = json.loads(match.group())
            scores = scores_json.get("scores", [])
        else:
            scores = []

    except Exception as e:
        print("Rerank error:", e)
        scores = []

    # Assign scores safely
    for i, candidate in enumerate(candidates):
        try:
            if i < len(scores):
                normalized = float(scores[i]) / 10.0
                candidate["score"] = max(0.0, min(1.0, normalized))
            else:
                candidate["score"] = 0.0
        except:
            candidate["score"] = 0.0

    return sorted(candidates, key=lambda x: x["score"], reverse=True)
