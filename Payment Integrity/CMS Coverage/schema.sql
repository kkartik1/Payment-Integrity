PRAGMA journal_mode=WAL;

-- Common document registry -----------------------------
CREATE TABLE IF NOT EXISTS documents (
  doc_pk INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_type TEXT CHECK(doc_type IN ('NCD','LCD','ARTICLE')) NOT NULL,
  doc_id TEXT NOT NULL,                  -- e.g., "220.12", "L34049", "A57183"
  title TEXT,
  status TEXT,                           -- in-effect / future-effective / retired
  effective_date TEXT,
  retire_date TEXT,
  version TEXT,
  last_updated TEXT,
  url TEXT,
  mac_id TEXT,                           -- contractor id if applicable
  mac_name TEXT,
  jurisdiction TEXT,                     -- state list or region text
  coverage_level TEXT,                   -- e.g., NCD cvrg_lvl_cd (1/2/3)
  UNIQUE(doc_type, doc_id)
);

-- Coverage narrative -----------------------------------
CREATE TABLE IF NOT EXISTS coverage_narrative (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  service_description TEXT,
  indications_limitations TEXT,
  documentation_requirements TEXT,
  frequency_limitations TEXT,
  related_documents TEXT                 -- JSON array of related IDs
);

-- Codes (normalized per document) ----------------------
CREATE TABLE IF NOT EXISTS doc_codes_hcpcs (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  code TEXT, version TEXT
);

CREATE TABLE IF NOT EXISTS doc_codes_icd10_covered (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  code TEXT, version TEXT
);

CREATE TABLE IF NOT EXISTS doc_codes_icd10_noncovered (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  code TEXT, version TEXT
);

CREATE TABLE IF NOT EXISTS doc_bill_type (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  code TEXT, version TEXT
);

CREATE TABLE IF NOT EXISTS doc_revenue_code (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  code TEXT, version TEXT, range_flag TEXT
);

CREATE TABLE IF NOT EXISTS doc_modifiers (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  code TEXT, version TEXT
);

-- Benefit Category for NCDs ----------------------------
CREATE TABLE IF NOT EXISTS ncd_benefit_category (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  bnft_ctgry_cd TEXT,
  bnft_ctgry_desc TEXT
);

-- Links between docs ----------------------------------
CREATE TABLE IF NOT EXISTS related_docs (
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  related_type TEXT,        -- 'NCD'|'LCD'|'ARTICLE'
  related_id TEXT           -- external id: e.g., "220.12" or "L12345"
);

-- Vector chunks metadata (text lives here; vectors in FAISS)
CREATE TABLE IF NOT EXISTS coverage_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_pk INTEGER NOT NULL REFERENCES documents(doc_pk) ON DELETE CASCADE,
  doc_type TEXT,
  doc_id TEXT,
  section TEXT,
  text TEXT,
  effective_date TEXT,
  mac_id TEXT,
  jurisdiction TEXT,
  hcpcs_json TEXT,
  icd10_json TEXT,
  bill_type_json TEXT,
  revenue_code_json TEXT
);

-- Index metadata for reproducibility
CREATE TABLE IF NOT EXISTS vector_index_meta (
  index_path TEXT PRIMARY KEY,
  model TEXT,
  dim INTEGER,
  total_vectors INTEGER,
  built_at TEXT
);