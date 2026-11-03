import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
PAGES_DIR = DATA_DIR / "pages"
TREES_DIR = DATA_DIR / "trees"
DB_PATH = DATA_DIR / "docs.sqlite3"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FAST_MODEL = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")
SMART_MODEL = os.getenv("SMART_MODEL", "openai/gpt-oss-120b")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "120"))
MAX_LEAF_PAGES = 4            # leaves longer than this are split into windows
WINDOW_PAGES = 3              # window size for splitting / fallback tree
SUMMARY_BATCH = 4             # nodes summarised per LLM call
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "14000"))
MAX_NODE_CHARS = 6000
OUTLINE_CHAR_BUDGET = 12000
MAX_NODES_SELECTED = 5
