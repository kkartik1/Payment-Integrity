import sqlite3, pandas as pd
from config import SQLITE_PATH, OUT_DIR

def main():
    with sqlite3.connect(SQLITE_PATH) as cx:
        # LCD related articles / NCDs
        lrel = pd.read_parquet(OUT_DIR / "LCD_LCD_RELATED_DOCUMENTS.parquet")
        for _, r in lrel.iterrows():
            src = f"L{int(r['lcd_id'])}"
            if pd.notna(r.get("r_article_id")):
                rid = f"A{int(r['r_article_id'])}"
                cx.execute("""INSERT INTO related_docs(doc_pk, related_type, related_id)
                              SELECT d.doc_pk, 'ARTICLE', ? FROM documents d WHERE d.doc_type='LCD' AND d.doc_id=?""",
                           (rid, src))
        lncd = pd.read_parquet(OUT_DIR / "LCD_LCD_RELATED_NCD_DOCUMENTS.parquet")
        for _, r in lncd.iterrows():
            if int(r["r_ncd_id"]) == 0: continue
            src = f"L{int(r['lcd_id'])}"
            rid = str(int(r["r_ncd_id"]))  # NCD uses manual section number
            cx.execute("""INSERT INTO related_docs(doc_pk, related_type, related_id)
                          SELECT d.doc_pk, 'NCD', ? FROM documents d WHERE d.doc_type='LCD' AND d.doc_id=?""",
                       (rid, src))
        # Article related LCD/NCD
        arel = pd.read_parquet(OUT_DIR / "ARTICLE_ARTICLE_RELATED_DOCUMENTS.parquet")
        for _, r in arel.iterrows():
            src = f"A{int(r['article_id'])}"
            if pd.notna(r.get("r_lcd_id")):
                rid = f"L{int(r['r_lcd_id'])}"
                cx.execute("""INSERT INTO related_docs(doc_pk, related_type, related_id)
                              SELECT d.doc_pk, 'LCD', ? FROM documents d WHERE d.doc_type='ARTICLE' AND d.doc_id=?""",
                           (rid, src))
        ancd = pd.read_parquet(OUT_DIR / "ARTICLE_ARTICLE_RELATED_NCD_DOCUMENTS.parquet")
        for _, r in ancd.iterrows():
            if int(r["r_ncd_id"]) == 0: continue
            src = f"A{int(r['article_id'])}"
            rid = str(int(r["r_ncd_id"]))
            cx.execute("""INSERT INTO related_docs(doc_pk, related_type, related_id)
                          SELECT d.doc_pk, 'NCD', ? FROM documents d WHERE d.doc_type='ARTICLE' AND d.doc_id=?""",
                       (rid, src))
        cx.commit()
    print("Built related-doc links.")

if __name__ == "__main__":
    main()