import argparse, sqlite3, json
import pandas as pd
from datetime import date
from config import SQLITE_PATH, VECTOR_DIR, EMBED_MODEL, TOP_K
from embeddings import VectorStore

def within_dates(eff, ret, dos):
    if eff and dos < eff: return False
    if ret and dos > ret: return False
    return True

def hard_filter(cx, cpt: str, icd10: str|None, dos: str, provider_state: str):
    # 1) Articles that list the CPT (+ optionally ICD10) and are in jurisdiction and effective on DOS
    q = """
    SELECT d.doc_pk, d.doc_id, d.doc_type, d.title, d.status, d.effective_date, d.retire_date, d.jurisdiction
    FROM documents d
    JOIN doc_codes_hcpcs h ON h.doc_pk = d.doc_pk
    LEFT JOIN doc_codes_icd10_covered i ON i.doc_pk = d.doc_pk
    WHERE d.doc_type IN ('ARTICLE','LCD')
      AND h.code = ?
    """
    #params = [cpt.strip()]
    
    if cpt is not None:
        params = [cpt.strip()]
    else:
        params = [""]

    rows = cx.execute(q, params).fetchall()
    
    # Jurisdiction & date screen
    ok = []
    for r in rows:
        eff = r[5]
        ret = r[6]
        juris = (r[7] or "")
        has_state = (provider_state in juris.split(",")) if juris else True
        dos_iso = dos
        if has_state and within_dates(eff, ret, dos_iso):
            if icd10:
                # If ICD provided, prefer docs that include it as covered (Articles usually carry ICD lists)
                icd_ok = cx.execute("SELECT 1 FROM doc_codes_icd10_covered WHERE doc_pk=? AND code=? LIMIT 1",
                                    (r[0], icd10)).fetchone()
                if not icd_ok and r[2] == 'ARTICLE':
                    continue
            ok.append(r[0])
    return list(dict.fromkeys(ok))  # unique doc_pks in order

def ann_rank(cx, doc_pks, cpt, icd10):
    if not doc_pks: return []
    # Build query text
    qtxt = f"Coverage for CPT {cpt}" + (f" with diagnosis {icd10}" if icd10 else "") + " - indications, limitations, documentation, frequency"
    
    vs = VectorStore.load(
        azure_endpoint="https://api.uhg.com/api/cloud/api-management/ai-gateway/1.0",
        api_key=None,  # or your API key if needed
        azure_deployment="text-embedding-3-small_1",
        api_version="2025-01-01-preview",
        index_path=VECTOR_DIR / "faiss.index"
    )

    # Retrieve top-N globally, then filter to chunks from doc_pks
    results = vs.search(qtxt, top_k=200)
    # Map chunk_id -> doc_pk
    placeholder = ",".join(["?"] * len(results))
    chunk_ids = [cid for cid, _ in results]
    cur = cx.cursor()
    cur.execute(f"SELECT chunk_id, doc_pk, section FROM coverage_chunks WHERE chunk_id IN ({placeholder})", chunk_ids)
    meta = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    ranked = []
    doc_pk_set = set(doc_pks)
    for cid, score in results:
        if cid in meta and meta[cid][0] in doc_pk_set:
            ranked.append((meta[cid][0], cid, meta[cid][1], score))
    # Take top-k distinct by doc
    seen, top = set(), []
    for doc_pk, cid, section, score in ranked:
        if doc_pk not in seen:
            seen.add(doc_pk)
            top.append((doc_pk, cid, section, score))
        if len(top) >= TOP_K: break
    return top

def assemble_context(cx, ranked):
    payload = []
    for doc_pk, cid, section, score in ranked:
        d = cx.execute("""SELECT doc_type, doc_id, title, status, effective_date, retire_date, jurisdiction
                          FROM documents WHERE doc_pk=?""", (doc_pk,)).fetchone()
        codes = {
          "hcpcs": [r[0] for r in cx.execute("SELECT code FROM doc_codes_hcpcs WHERE doc_pk=?", (doc_pk,)).fetchall()],
          "icd10": [r[0] for r in cx.execute("SELECT code FROM doc_codes_icd10_covered WHERE doc_pk=?", (doc_pk,)).fetchall()],
          "bill_type": [r[0] for r in cx.execute("SELECT code FROM doc_bill_type WHERE doc_pk=?", (doc_pk,)).fetchall()],
          "revenue": [r[0] for r in cx.execute("SELECT code FROM doc_revenue_code WHERE doc_pk=?", (doc_pk,)).fetchall()],
        }
        text, = cx.execute("SELECT text FROM coverage_chunks WHERE chunk_id=?", (cid,)).fetchone()
        payload.append({
          "score": score,
          "doc": {"type": d[0], "id": d[1], "title": d[2], "status": d[3],
                  "effective_date": d[4], "retire_date": d[5], "jurisdiction": d[6]},
          "section": section,
          "excerpt": text[:1000],
          "codes": codes
        })
    return payload


def main():
    # Connect to claims database
    claims_db_path = '/app/users/kkartik2/source/claims_database.db'
    with sqlite3.connect(SQLITE_PATH) as cx, sqlite3.connect(claims_db_path) as claims_cx:
        claims_cur = claims_cx.cursor()
        claims_cur.execute("SELECT Diagnosis_Code_1, Diagnosis_Code_2, Diagnosis_Code_3, Diagnosis_Code_4, Diagnosis_Code_5, Diagnosis_Code_6, Diagnosis_Code_7, Diagnosis_Code_8, Primary_Diagnosis_Code, Procedure_Code_ID, Line_Service_Start_Date, Service_Provider_State, Claim_ID FROM ClaimsData LIMIT 500")
        rows = claims_cur.fetchall()

        results = []
        for row in rows:
            diag_codes = row[:9]
            cpt = row[9]
            dos = row[10]
            state = row[11]
            claim_id = row[12]
            # For each non-empty ICD10 code, run the query logic
            for icd10 in diag_codes:
                if icd10 and icd10.strip():
                    doc_pks = hard_filter(cx, cpt, icd10, dos, state.upper() if state else "")
                    print(doc_pks)
                    ranked = ann_rank(cx, doc_pks, cpt, icd10)
                    out = assemble_context(cx, ranked)
                    for entry in out:
                        entry_flat = {
                            "Claim_ID": claim_id,
                            "CPT": cpt,
                            "ICD10": icd10,
                            "DOS": dos,
                            "State": state,
                            "Score": entry.get("score"),
                            "Doc_Type": entry["doc"].get("type"),
                            "Doc_ID": entry["doc"].get("id"),
                            "Title": entry["doc"].get("title"),
                            "Status": entry["doc"].get("status"),
                            "Effective_Date": entry["doc"].get("effective_date"),
                            "Retire_Date": entry["doc"].get("retire_date"),
                            "Jurisdiction": entry["doc"].get("jurisdiction"),
                            "Section": entry.get("section"),
                            "Excerpt": entry.get("excerpt"),
                            "Codes": json.dumps(entry.get("codes"))
                        }
                        results.append(entry_flat)

        # Write results to CSV
        df = pd.DataFrame(results)
        df.to_csv("output_claims_query_results.csv", index=False)



if __name__ == "__main__":
    main()