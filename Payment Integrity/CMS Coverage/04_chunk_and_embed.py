import json, sqlite3, uuid
from tqdm import tqdm
from config import SQLITE_PATH, VECTOR_DIR, EMBED_MODEL, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS
from html_utils import strip_html
from chunking import split_sections
from embeddings import VectorStore

def build_chunks_for_doc(cur, doc_pk):
    # Fetch narrative + codes
    cur.execute("SELECT doc_type, doc_id, effective_date, jurisdiction FROM documents WHERE doc_pk=?", (doc_pk,))
    doc_type, doc_id, eff, juris = cur.fetchone()
    cur.execute("""SELECT service_description, indications_limitations, documentation_requirements, frequency_limitations
                   FROM coverage_narrative WHERE doc_pk=?""", (doc_pk,))
    row = cur.fetchone() or ("","","","")
    service_description, indications, docreq, freq = [strip_html(x) for x in row]
    sections = []
    if service_description: sections.append(("Service Description", service_description))
    if indications: sections.append(("Coverage Indications & Limitations", indications))
    if docreq: sections.append(("Documentation Requirements", docreq))
    if freq: sections.append(("Frequency Limitations", freq))

    # Pull code lists for metadata
    def list_codes(table):
        cur.execute(f"SELECT code FROM {table} WHERE doc_pk=?", (doc_pk,))
        return [r[0] for r in cur.fetchall()]
    hcpcs = list_codes("doc_codes_hcpcs")
    icd10_cov = list_codes("doc_codes_icd10_covered")
    bill = list_codes("doc_bill_type")
    rev = list_codes("doc_revenue_code")

    chunks = []
    for section_name, chunk_text in split_sections(sections, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS):
        chunk_id = str(uuid.uuid4())
        cur.execute("""INSERT OR REPLACE INTO coverage_chunks
                       (chunk_id, doc_pk, doc_type, doc_id, section, text, effective_date, mac_id, jurisdiction,
                        hcpcs_json, icd10_json, bill_type_json, revenue_code_json)
                       SELECT ?, d.doc_pk, d.doc_type, d.doc_id, ?, ?, d.effective_date, d.mac_id, d.jurisdiction,
                        ?, ?, ?, ?
                       FROM documents d WHERE d.doc_pk=?""",
                    (chunk_id, section_name, chunk_text, json.dumps(hcpcs), json.dumps(icd10_cov),
                     json.dumps(bill), json.dumps(rev), doc_pk))
        chunks.append((chunk_id, chunk_text))
    return chunks

def main():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    # Update these parameters as per your Azure deployment
    azure_endpoint = "https://api.uhg.com/api/cloud/api-management/ai-gateway/1.0"
    api_key = None  # Not used for Azure AD token flow
    azure_deployment = "text-embedding-3-small_1"
    api_version = "2025-01-01-preview"

    vs = VectorStore(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        azure_deployment=azure_deployment,
        api_version=api_version,
        index_path=VECTOR_DIR / "faiss.index"
    )
    with sqlite3.connect(SQLITE_PATH) as cx:
        cur = cx.cursor()
        cur.execute("SELECT doc_pk FROM documents")
        doc_pks = [r[0] for r in cur.fetchall()]
        all_pairs = []
        for doc_pk in tqdm(doc_pks, desc="Chunking"):
            pairs = build_chunks_for_doc(cur, doc_pk)
            all_pairs.extend(pairs)
        cx.commit()
        # Embed & save index
        vs.add(all_pairs)
        vs.save()
        print("Vector index built.")

if __name__ == "__main__":
    main()