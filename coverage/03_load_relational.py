import json, sqlite3
from pathlib import Path
import pandas as pd
from dateutil import parser as du
from config import OUT_DIR, SQLITE_PATH

def parse_date(s):
    try: return du.parse(str(s)).date().isoformat()
    except: return None

def load_documents(cx):
    # ---------- NCD ----------
    ncd = pd.read_parquet(OUT_DIR / "NCD_NCD_TRKG.parquet")
    ncd["effective_date"] = ncd["NCD_efctv_dt"].apply(parse_date)
    ncd["retire_date"] = ncd["NCD_trmntn_dt"].apply(parse_date)
    ncd["status"] = ncd.apply(lambda r: "retired" if pd.notna(r.get("NCD_trmntn_dt")) else "in-effect", axis=1)
    ncd["title"] = ncd["NCD_mnl_sect_title"]
    ncd["version"] = ncd["NCD_vrsn_num"].astype(str)
    ncd["last_updated"] = ncd["last_updt_tmstmp"].astype(str)
    ncd["coverage_level"] = ncd["cvrg_lvl_cd"].astype(str)  # 1/2/3 as per dictionary
    records = []
    for _, r in ncd.iterrows():
        records.append((
            "NCD", str(r["NCD_mnl_sect"]), r["title"], r["status"], r["effective_date"], r["retire_date"],
            r["version"], r["last_updated"], r.get("trnsmtl_url"), None, None, None, r["coverage_level"]
        ))
    cx.executemany("""
      INSERT OR IGNORE INTO documents
      (doc_type, doc_id, title, status, effective_date, retire_date, version, last_updated, url, mac_id, mac_name, jurisdiction, coverage_level)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)

    # Benefit categories
    xref = pd.read_parquet(OUT_DIR / "NCD_NCD_TRKG_BNFT_XREF.parquet")
    ref = pd.read_parquet(OUT_DIR / "NCD_NCD_BNFT_CTGRY_REF.parquet")
    bn = xref.merge(ref, left_on="bnft_ctgry_cd", right_on="bnft_ctgry_cd", how="left")
    for _, r in bn.iterrows():
        doc_id = str(ncd.loc[ncd["NCD_id"]==r["NCD_id"], "NCD_mnl_sect"].iloc[0])
        (doc_pk,) = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='NCD' AND doc_id=?", (doc_id,)).fetchone()
        cx.execute("INSERT INTO ncd_benefit_category(doc_pk, bnft_ctgry_cd, bnft_ctgry_desc) VALUES (?,?,?)",
                   (doc_pk, r["bnft_ctgry_cd"], r["bnft_ctgry_desc"]))

    # Narrative for NCD
    for _, r in ncd.iterrows():
        doc_id = str(r["NCD_mnl_sect"])
        (doc_pk,) = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='NCD' AND doc_id=?", (doc_id,)).fetchone()
        cx.execute("""INSERT OR REPLACE INTO coverage_narrative
                      (doc_pk, service_description, indications_limitations, documentation_requirements, frequency_limitations, related_documents)
                      VALUES (?,?,?,?,?,?)""",
                   (doc_pk, r.get("itm_srvc_desc"), r.get("indctn_lmtn"), None, None, None))

    # ---------- LCD ----------
    lcd = pd.read_parquet(OUT_DIR / "LCD_LCD.parquet")
    lcd["effective_date"] = lcd["orig_det_eff_date"].apply(parse_date).fillna(lcd["rev_eff_date"].apply(parse_date))
    lcd["retire_date"] = lcd["date_retired"].apply(parse_date).fillna(lcd["ent_det_end_date"].apply(parse_date))
    def lcd_status(row):
        s = (row.get("status") or "").strip().upper()
        if s == "R": return "retired"
        return "in-effect"
    lcd["status_mapped"] = lcd.apply(lcd_status, axis=1)
    for _, r in lcd.iterrows():
        cx.execute("""
          INSERT OR IGNORE INTO documents
          (doc_type, doc_id, title, status, effective_date, retire_date, version, last_updated, url, mac_id, mac_name, jurisdiction, coverage_level)
          VALUES ('LCD', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
        """, (f"L{int(r['lcd_id'])}", r["title"], r["status_mapped"], r["effective_date"], r["retire_date"],
              str(r["lcd_version"]), str(r["last_updated"])))

        (doc_pk,) = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='LCD' AND doc_id=?", (f"L{int(r['lcd_id'])}",)).fetchone()
        cx.execute("""INSERT OR REPLACE INTO coverage_narrative
                      (doc_pk, service_description, indications_limitations, documentation_requirements, frequency_limitations, related_documents)
                      VALUES (?,?,?,?,?,?)""",
                   (doc_pk, None, r.get("indication"), r.get("doc_reqs"), r.get("util_guide"), None))

    # LCD codes (HCPCS)
    lcd_h = pd.read_parquet(OUT_DIR / "LCD_LCD_X_HCPC_CODE.parquet")
    for _, r in lcd_h.iterrows():
        doc_id = f"L{int(r['lcd_id'])}"
        row = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='LCD' AND doc_id=?", (doc_id,)).fetchone()
        if not row: continue
        (doc_pk,) = row
        cx.execute("INSERT INTO doc_codes_hcpcs(doc_pk, code, version) VALUES (?,?,?)",
                   (doc_pk, str(r["hcpc_code_id"]), str(r["hcpc_code_version"])))

    # LCD jurisdiction/contractor
    lcd_contra = pd.read_parquet(OUT_DIR / "LCD_LCD_X_CONTRACTOR.parquet")
    con = pd.read_parquet(OUT_DIR / "LCD_CONTRACTOR.parquet")
    cj = pd.read_parquet(OUT_DIR / "LCD_CONTRACTOR_JURISDICTION.parquet")
    states = pd.read_parquet(OUT_DIR / "LCD_STATE_LOOKUP.parquet")[["state_id","state_abbrev"]].rename(columns={"state_abbrev":"abbr"})
    for _, r in lcd_contra.iterrows():
        doc_id = f"L{int(r['lcd_id'])}"
        row = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='LCD' AND doc_id=?", (doc_id,)).fetchone()
        if not row: continue
        (doc_pk,) = row
        c = con[(con.contractor_id==r.contractor_id)&(con.contractor_type_id==r.contractor_type_id)&(con.contractor_version==r.contractor_version)]
        mac_name = c.contractor_bus_name.iloc[0] if len(c) else None
        mac_id = f"{int(r.contractor_id)}-{int(r.contractor_type_id)}-{int(r.contractor_version)}"
        states_for = cj[(cj.contractor_id==r.contractor_id)&(cj.contractor_type_id==r.contractor_type_id)&(cj.contractor_version==r.contractor_version)]
        abbrs = states_for.merge(states, on="state_id", how="left").abbr.dropna().unique().tolist()
        cx.execute("UPDATE documents SET mac_id=?, mac_name=?, jurisdiction=? WHERE doc_pk=?",
                   (mac_id, mac_name, ",".join(abbrs), doc_pk))

    # ---------- ARTICLE ----------
    art = pd.read_parquet(OUT_DIR / "ARTICLE_ARTICLE.parquet")
    art["effective_date"] = art["article_eff_date"].apply(parse_date)
    art["retire_date"] = art["article_end_date"].apply(parse_date).fillna(art["date_retired"].apply(parse_date))
    def art_status(row):
        s = (row.get("status") or "").strip().upper()
        if s == "R": return "retired"
        return "in-effect"
    art["status_mapped"] = art.apply(art_status, axis=1)
    for _, r in art.iterrows():
        doc_id = f"A{int(r['article_id'])}"
        cx.execute("""
          INSERT OR IGNORE INTO documents
          (doc_type, doc_id, title, status, effective_date, retire_date, version, last_updated, url, mac_id, mac_name, jurisdiction, coverage_level)
          VALUES ('ARTICLE', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
        """, (doc_id, r["title"], r["status_mapped"], r["effective_date"], r["retire_date"],
              str(r["article_version"]), str(r["last_updated"])))
        (doc_pk,) = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='ARTICLE' AND doc_id=?", (doc_id,)).fetchone()
        cx.execute("""INSERT OR REPLACE INTO coverage_narrative
                      (doc_pk, service_description, indications_limitations, documentation_requirements, frequency_limitations, related_documents)
                      VALUES (?,?,?,?,?,?)""",
                   (doc_pk, None, r.get("description"), None, None, None))

    # Article codes: HCPCS/ICD/Bill/Revenue/Modifiers
    def insert_codes(tbl_name, dest, code_col, ver_col):
        df = pd.read_parquet(OUT_DIR / f"ARTICLE_{tbl_name}.parquet")
        for _, r in df.iterrows():
            doc_id = f"A{int(r['article_id'])}"
            row = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='ARTICLE' AND doc_id=?", (doc_id,)).fetchone()
            if not row: continue
            (doc_pk,) = row
            code = str(r[code_col])
            ver = str(r.get(ver_col)) if ver_col in r else None
            cx.execute(f"INSERT INTO {dest}(doc_pk, code, version) VALUES (?,?,?)", (doc_pk, code, ver))
    insert_codes("ARTICLE_X_HCPC_CODE", "doc_codes_hcpcs", "hcpc_code_id", "hcpc_code_version")
    insert_codes("ARTICLE_X_ICD10_COVERED", "doc_codes_icd10_covered", "icd10_code_id", "icd10_code_version")
    insert_codes("ARTICLE_X_ICD10_NONCOVERED", "doc_codes_icd10_noncovered", "icd10_code_id", "icd10_code_version")
    insert_codes("ARTICLE_X_BILL_CODE", "doc_bill_type", "bill_code_id", "bill_code_version")
    # revenue codes include "range"
    rev = pd.read_parquet(OUT_DIR / "ARTICLE_ARTICLE_X_REVENUE_CODE.parquet")
    for _, r in rev.iterrows():
        doc_id = f"A{int(r['article_id'])}"
        row = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='ARTICLE' AND doc_id=?", (doc_id,)).fetchone()
        if not row: continue
        (doc_pk,) = row
        cx.execute("INSERT INTO doc_revenue_code(doc_pk, code, version, range_flag) VALUES (?,?,?,?)",
                   (doc_pk, str(r["revenue_code_id"]), str(r["revenue_code_version"]), str(r["range"])))
    # modifiers
    mod = pd.read_parquet(OUT_DIR / "ARTICLE_ARTICLE_X_HCPC_MODIFIER.parquet")
    for _, r in mod.iterrows():
        doc_id = f"A{int(r['article_id'])}"
        row = cx.execute("SELECT doc_pk FROM documents WHERE doc_type='ARTICLE' AND doc_id=?", (doc_id,)).fetchone()
        if not row: continue
        (doc_pk,) = row
        cx.execute("INSERT INTO doc_modifiers(doc_pk, code, version) VALUES (?,?,?)",
                   (doc_pk, str(r["hcpc_modifier_code_id"]), str(r["hcpc_modifier_code_version"])))

def main():
    with sqlite3.connect(SQLITE_PATH) as cx:
        cx.execute("PRAGMA foreign_keys=ON")
        load_documents(cx)
        cx.commit()
    print("Loaded relational layer.")

if __name__ == "__main__":
    main()