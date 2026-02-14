from flask import Flask, request, jsonify
from search import semantic_search

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    result = semantic_search(data)
    return jsonify(result)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(port=5000)
