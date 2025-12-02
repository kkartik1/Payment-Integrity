import subprocess, shutil
from io import StringIO
import pandas as pd

def require_mdbtools():
    if shutil.which("mdb-tables") is None or shutil.which("mdb-export") is None:
        raise RuntimeError("mdbtools required (install via apt/dnf).")

def list_tables(mdb_path: str):
    out = subprocess.run(["mdb-tables", "-1", mdb_path], capture_output=True, text=True, check=True).stdout
    return [t.strip() for t in out.splitlines() if t.strip()]

def read_table(mdb_path: str, table: str) -> pd.DataFrame:
    # mdb-export emits CSV; we parse via pandas
    out = subprocess.run(["mdb-export", mdb_path, table], capture_output=True, text=True, check=True).stdout
    if not out.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(out))