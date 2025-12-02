from pathlib import Path
import pandas as pd
from mdb_utils import require_mdbtools, read_table
from config import MDB, OUT_DIR

NEEDED = {
    "NCD": ["NCD_TRKG", "NCD_TRKG_BNFT_XREF", "NCD_BNFT_CTGRY_REF", "NCD_PBLCTN_REF"],
    "LCD": ["LCD", "LCD_X_HCPC_CODE", "LCD_X_HCPC_CODE_GROUP", "LCD_RELATED_DOCUMENTS",
            "LCD_RELATED_NCD_DOCUMENTS", "LCD_FUTURE_RETIRE", "LCD_X_CONTRACTOR",
            "CONTRACTOR", "CONTRACTOR_JURISDICTION", "LCD_X_PRIMARY_JURISDICTION", "STATE_LOOKUP"],
    "ARTICLE": ["ARTICLE", "ARTICLE_X_HCPC_CODE", "ARTICLE_X_ICD10_COVERED",
                "ARTICLE_X_ICD10_NONCOVERED", "ARTICLE_X_BILL_CODE", "ARTICLE_X_REVENUE_CODE",
                "ARTICLE_X_HCPC_MODIFIER", "ARTICLE_X_HCPC_CODE_GROUP",
                "ARTICLE_RELATED_DOCUMENTS", "ARTICLE_RELATED_NCD_DOCUMENTS",
                "ARTICLE_X_CONTRACTOR", "CONTRACTOR", "CONTRACTOR_JURISDICTION",
                "ARTICLE_X_PRIMARY_JURISDICTION", "STATE_LOOKUP"]
}

def main():
    require_mdbtools()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, mdb in MDB.items():
        for tbl in NEEDED[key]:
            print(f"[{key}] reading {tbl}")
            df = read_table(str(mdb), tbl)
            df.to_parquet(OUT_DIR / f"{key}_{tbl}.parquet", index=False)

if __name__ == "__main__":
    main()