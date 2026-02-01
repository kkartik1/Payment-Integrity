from pathlib import Path

DATA_DIR = Path("data")
MDB = {
    "NCD": DATA_DIR / "ncd.mdb",
    "LCD": DATA_DIR / "current_lcd.mdb",
    "ARTICLE": DATA_DIR / "current_article.mdb",
}

OUT_DIR = Path("out")
SQLITE_PATH = OUT_DIR / "cms.sqlite"
VECTOR_DIR = OUT_DIR / "vectors"

# Embeddings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small, fast
CHUNK_MIN_TOKENS = 500
CHUNK_MAX_TOKENS = 1200
TOP_K = 20