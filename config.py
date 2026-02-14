import os
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RERANK_MODEL = os.getenv("RERANK_MODEL")

HEADERS = {
    "Authorization": f"Bearer {AIPIPE_TOKEN}",
    "Content-Type": "application/json"
}
