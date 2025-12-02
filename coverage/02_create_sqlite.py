import sqlite3
from config import SQLITE_PATH
from pathlib import Path

def main():
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ddl = Path("schema.sql").read_text()
    with sqlite3.connect(SQLITE_PATH) as cx:
        cx.executescript(ddl)
    print(f"Created {SQLITE_PATH}")

if __name__ == "__main__":
    main()