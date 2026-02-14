from flask import Flask, request, jsonify
from search import semantic_search
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    result = semantic_search(data)
    return jsonify(result)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/documents")
def documents():
    return {"message": "Documents loaded"}



import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
