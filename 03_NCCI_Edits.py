
"""
NCCI Validation – SQLite Claims Runner
-------------------------------------
This script extends the provided `NCCI (1).py` by reading claims and claim lines
from a SQLite database (table `ClaimsData`) and writing out a CSV of flags with
clear reasons at claim/claim-line level.

Key capabilities
* Connects to SQLite DB: `--db claims_database.db`
* Pulls required fields from `ClaimsData`
* Groups rows by Claim_ID, builds line items (HCPCS + units + modifiers + DOS)
* Runs PTP & MUE checks using NCCIAgent
* Adds **aggregated MUE checks for MAI=2 (claim-level)** and **MAI=3 (date-of-service)**
* Outputs `--out ncci_validation_results.csv` with detailed reasons

Usage
-----
python ncci_sqlite.py --db claims_database.db --out ncci_validation_results.csv --auto-fetch

If you need to pin to a specific quarter/year:
python ncci_sqlite.py --db claims_database.db --year 2025 --quarter 4 --auto-fetch

Note: If your environment has no internet access, omit `--auto-fetch` to use the
sample NCCI data embedded in the agent (good for plumbing tests only).
"""
import argparse
import csv
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# --- Begin: NCCIAgent code (lightly adapted from the provided file) ---
import json
import requests
import zipfile
import io
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EditType(Enum):
    UNBUNDLING = "unbundling"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    MUE_EXCEEDED = "mue_exceeded"

class ServiceType(Enum):
    PRACTITIONER = "PRA"
    HOSPITAL = "OPH"
    DME = "DME"

@dataclass
class NCCIResource:
    uri: str
    description: str

@dataclass
class PTPEdit:
    column1_code: str
    column2_code: str
    modifier_indicator: str
    effective_date: str
    deletion_date: Optional[str]
    edit_type: str
    can_bypass_with_modifier: bool

@dataclass
class MUEEdit:
    hcpcs_code: str
    mue_value: int
    mue_adjudication_indicator: str
    units_billed: int
    exceeds_limit: bool
    effective_date: str

@dataclass
class ValidationResult:
    valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    ptp_edits: List[PTPEdit]
    mue_edits: List[MUEEdit]
    date_of_service: str

class CMSDataFetcher:
    BASE_URL = "https://www.cms.gov"
    PTP_PAGE = "/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-procedure-procedure-ptp-edits"
    MUE_PAGE = "/medicare/coding-billing/ncci-medicare/ncci-medicare-medically-unlikely-edits"
    PTP_URL_PATTERN = "/files/zip/medicare-ncci-{quarter}-{service_type}-ptp-edits-cci{service_code}-v{version}-f{file_num}.zip"
    MUE_URL_PATTERN = "/files/zip/medicare-ncci-{quarter}-{service_type}-services-mue-table.zip"

    def __init__(self, cache_dir: str = "./ncci_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'NCCI-Validation-Agent/1.0'})

    def get_current_quarter(self) -> Tuple[int, int]:
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        return now.year, quarter

    def build_ptp_url(self, year: int, quarter: int, service_type: str = "practitioner",
                       version: str = "313r0", file_num: int = 1) -> str:
        service_code = "pra" if service_type == "practitioner" else "oph"
        quarter_str = f"{year}q{quarter}"
        url = self.BASE_URL + self.PTP_URL_PATTERN.format(
            quarter=quarter_str,
            service_type=service_type,
            service_code=service_code,
            version=version,
            file_num=file_num
        )
        return url

    def build_mue_url(self, year: int, quarter: int, service_type: str = "practitioner") -> str:
        quarter_str = f"{year}-q{quarter}"
        url = self.BASE_URL + self.MUE_URL_PATTERN.format(
            quarter=quarter_str,
            service_type=service_type
        )
        return url

    def download_and_extract_zip(self, url: str) -> Dict[str, bytes]:
        logger.info(f"Downloading: {url}")
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            files = {}
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for filename in zf.namelist():
                    files[filename] = zf.read(filename)
                    logger.info(f" Extracted: {filename}")
            return files
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return {}

    def parse_ptp_file(self, file_content: bytes) -> List[Dict]:
        edits: List[Dict] = []
        try:
            excel_buffer = io.BytesIO(file_content)
            df = pd.read_excel(excel_buffer, engine='openpyxl', header=None, dtype=str)
            df.columns = ['Column 1', 'Column 2', 'Ignore', 'Effective Date', 'Deletion Date', 'Modifier', 'PTP Edit Rationale']
            df = df.iloc[6:]
            df = df[['Column 1', 'Column 2', 'Effective Date', 'Deletion Date', 'Modifier', 'PTP Edit Rationale']]
            df = df.dropna(subset=['Column 1', 'Column 2'], how='all')
            df = df.apply(lambda col: col.map(lambda x: str(x).strip('\ufeff').strip() if pd.notnull(x) else ''))
            edits = [
                {
                    'column1': row['Column 1'],
                    'column2': row['Column 2'],
                    'effective_date': row['Effective Date'],
                    'deletion_date': row['Deletion Date'],
                    'modifier_indicator': row['Modifier'],
                    'rationale': row['PTP Edit Rationale']
                }
                for _, row in df.iterrows()
            ]
            logger.info(f"Parsed {len(edits)} PTP edits from Excel file.")
        except Exception as e:
            logger.error(f"Failed to parse PTP file: {e}")
        return edits

    def parse_mue_file(self, file_content: bytes) -> Dict[str, Dict]:
        import csv as _csv
        mues = {}
        try:
            text = file_content.decode('utf-8', errors='ignore')
            lines = text.splitlines()
            header_index = None
            for i, line in enumerate(lines):
                if "HCPCS/" in line:
                    header_index = i
                    break
            if header_index is None or header_index + 1 >= len(lines):
                raise ValueError("Header row with expected columns not found.")
            header_line = lines[header_index].replace('"', '').replace('/', '').strip() + lines[header_index + 1].strip()
            header_line = header_line.replace(',,', ',').replace('",', ',').replace('CPT Code', 'CPT Code')
            data_lines = [header_line] + lines[header_index + 2:]
            reader = _csv.DictReader(io.StringIO('\n'.join(data_lines)), delimiter=',')
            def find_col(possibilities):
                for col in reader.fieldnames:
                    for p in possibilities:
                        if p.lower() in col.lower():
                            return col
                return None
            hcpcs_col = find_col(['HCPCS', 'CPT Code'])
            mue_val_col = find_col(['MUE Value'])
            mai_col = find_col(['Adjudication Indicator'])
            rationale_col = find_col(['Rationale'])
            eff_date_col = find_col(['Effective Date'])
            for row in reader:
                cleaned_row = {k.strip('\ufeff').strip(): (v.strip() if v is not None else '') for k, v in row.items()}
                hcpcs = cleaned_row.get(hcpcs_col, '').strip() if hcpcs_col else ''
                if not hcpcs:
                    continue
                try:
                    mue_value = int(cleaned_row.get(mue_val_col, '0')) if mue_val_col else 0
                except (ValueError, TypeError):
                    mue_value = 0
                mues[hcpcs] = {
                    'mue': mue_value,
                    'mai': cleaned_row.get(mai_col, '') if mai_col else '',
                    'rationale': cleaned_row.get(rationale_col, '') if rationale_col else '',
                    'effective_date': cleaned_row.get(eff_date_col, '') if eff_date_col else ''
                }
            logger.info(f"Parsed {len(mues)} MUE edits")
        except Exception as e:
            logger.error(f"Failed to parse MUE file: {e}")
        return mues

    def fetch_ptp_edits(self, year: int = None, quarter: int = None, service_type: str = "practitioner") -> List[Dict]:
        if year is None or quarter is None:
            year, quarter = self.get_current_quarter()
        all_edits = []
        for file_num in range(1, 5):
            url = self.build_ptp_url(year, quarter, service_type, file_num=file_num)
            files = self.download_and_extract_zip(url)
            if not files:
                continue
            for filename, content in files.items():
                if filename.endswith('.xlsx'):
                    edits = self.parse_ptp_file(content)
                    all_edits.extend(edits)
                    break
        return all_edits

    def fetch_mue_edits(self, year: int = None, quarter: int = None, service_type: str = "practitioner") -> Dict[str, Dict]:
        if year is None or quarter is None:
            year, quarter = self.get_current_quarter()
        all_edits: Dict[str, Dict] = {}
        service_type_lst = ["practitioner", "dme-supplier", "facility-outpatient-hospital"]
        for st in service_type_lst:
            url = self.build_mue_url(year, quarter, st)
            files = self.download_and_extract_zip(url)
            if not files:
                continue
            for filename, content in files.items():
                if filename.endswith('.csv'):
                    edits = self.parse_mue_file(content)
                    all_edits.update(edits)
                    break
        return all_edits

class NCCIAgent:
    def __init__(self, config: Dict[str, Any], auto_fetch: bool = False, year: Optional[int] = None, quarter: Optional[int] = None):
        self.name = config.get("name", "cms-ncci")
        self.resources = [NCCIResource(**r) for r in config.get("resources", [])]
        self.tools = config.get("tools", [])
        self.ptp_data: Dict[str, List[Dict]] = {}
        self.mue_data: Dict[str, Dict] = {}
        self.fetcher = CMSDataFetcher()
        if auto_fetch:
            self.fetch_latest_data(year, quarter)
        else:
            self._load_sample_data()

    def fetch_latest_data(self, year: int = None, quarter: int = None):
        logger.info("Fetching latest NCCI data from CMS...")
        ptp_edits = self.fetcher.fetch_ptp_edits(year, quarter, "practitioner")
        self._load_ptp_data(ptp_edits)
        mue_edits = self.fetcher.fetch_mue_edits(year, quarter, "practitioner")
        self._load_mue_data(mue_edits)
        logger.info(f"Loaded {len(self.ptp_data)} PTP column1 codes")
        logger.info(f"Loaded {len(self.mue_data)} MUE codes")

    def _load_ptp_data(self, ptp_edits: List[Dict]):
        self.ptp_data = {}
        for edit in ptp_edits:
            col1 = edit['column1']
            if col1 not in self.ptp_data:
                self.ptp_data[col1] = []
            rationale = edit.get('rationale', '').lower()
            edit_type = "mutually_exclusive" if "mutually" in rationale else "unbundling"
            self.ptp_data[col1].append({
                'column2': edit['column2'],
                'modifier_indicator': edit['modifier_indicator'],
                'effective_date': edit['effective_date'].replace('-', ''),
                'deletion_date': edit['deletion_date'].replace('-', '') if edit['deletion_date'] else None,
                'edit_type': edit_type
            })

    def _load_mue_data(self, mue_edits: Dict[str, Dict]):
        self.mue_data = {}
        for hcpcs, edit in mue_edits.items():
            self.mue_data[hcpcs] = {
                'mue': edit.get('mue', 0),
                'mai': edit.get('mai', ''),
                'rationale': edit.get('rationale', ''),
                'effective_date': edit.get('effective_date', '')
            }

    def _load_sample_data(self):
        logger.info("Loading sample NCCI data (for offline/demo only)...")
        self.ptp_data = {
            "99213": [
                {"column2": "36415", "modifier_indicator": "1", "effective_date": "20250101", "deletion_date": None, "edit_type": "unbundling"}
            ],
            "27447": [
                {"column2": "27369", "modifier_indicator": "0", "effective_date": "20250101", "deletion_date": None, "edit_type": "unbundling"}
            ],
            "29881": [
                {"column2": "29880", "modifier_indicator": "1", "effective_date": "20250101", "deletion_date": None, "edit_type": "mutually_exclusive"}
            ]
        }
        self.mue_data = {
            "99213": {"mue": 4, "mai": "2", "effective_date": "20250101"},
            "36415": {"mue": 2, "mai": "2", "effective_date": "20250101"},
            "27447": {"mue": 1, "mai": "1", "effective_date": "20250101"},
            "J0135": {"mue": 50, "mai": "2", "effective_date": "20250101"},
            "80053": {"mue": 1, "mai": "3", "effective_date": "20250101"},
        }

    def ptpcheck(self, hcpcs: List[str], dos: str) -> ValidationResult:
        errors = []
        warnings = []
        ptp_edits = []
        dos_date = datetime.strptime(dos, "%Y%m%d")
        for i, code1 in enumerate(hcpcs):
            for code2 in hcpcs[i+1:]:
                edit = self._check_ptp_pair(code1, code2, dos_date)
                if edit:
                    ptp_edits.append(edit)
                    if edit.can_bypass_with_modifier:
                        warnings.append({
                            "type": edit.edit_type,
                            "message": f"PTP edit: {code1} bundles {code2}. May bypass with modifier.",
                            "codes": [code1, code2],
                            "modifier_indicator": edit.modifier_indicator
                        })
                    else:
                        errors.append({
                            "type": edit.edit_type,
                            "message": f"PTP edit: {code1} bundles {code2}. Cannot bypass.",
                            "codes": [code1, code2],
                            "modifier_indicator": edit.modifier_indicator
                        })
                edit_rev = self._check_ptp_pair(code2, code1, dos_date)
                if edit_rev:
                    ptp_edits.append(edit_rev)
                    if edit_rev.can_bypass_with_modifier:
                        warnings.append({
                            "type": edit_rev.edit_type,
                            "message": f"PTP edit: {code2} bundles {code1}. May bypass with modifier.",
                            "codes": [code2, code1],
                            "modifier_indicator": edit_rev.modifier_indicator
                        })
                    else:
                        errors.append({
                            "type": edit_rev.edit_type,
                            "message": f"PTP edit: {code2} bundles {code1}. Cannot bypass.",
                            "codes": [code2, code1],
                            "modifier_indicator": edit_rev.modifier_indicator
                        })
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            ptp_edits=ptp_edits,
            mue_edits=[],
            date_of_service=dos
        )

    def _check_ptp_pair(self, column1: str, column2: str, dos_date: datetime) -> Optional[PTPEdit]:
        if column1 not in self.ptp_data:
            return None
        for edit in self.ptp_data[column1]:
            if edit["column2"] == column2:
                effective_date = datetime.strptime(edit["effective_date"], "%Y%m%d")
                deletion_date = datetime.strptime(edit["deletion_date"], "%Y%m%d") if (edit["deletion_date"] not in ['', '*', None]) else None
                if effective_date <= dos_date and (deletion_date is None or dos_date < deletion_date):
                    return PTPEdit(
                        column1_code=column1,
                        column2_code=column2,
                        modifier_indicator=edit["modifier_indicator"],
                        effective_date=edit["effective_date"],
                        deletion_date=edit["deletion_date"],
                        edit_type=edit["edit_type"],
                        can_bypass_with_modifier=(edit["modifier_indicator"] == "1")
                    )
        return None

    def muecheck(self, hcpcs: str, uom: int, dos: str) -> ValidationResult:
        errors = []
        warnings = []
        mue_edits = []
        if hcpcs in self.mue_data:
            mue_info = self.mue_data[hcpcs]
            mue_value = mue_info["mue"]
            mai = mue_info["mai"]
            exceeds = uom > mue_value
            mue_edit = MUEEdit(
                hcpcs_code=hcpcs,
                mue_value=mue_value,
                mue_adjudication_indicator=mai,
                units_billed=uom,
                exceeds_limit=exceeds,
                effective_date=mue_info.get("effective_date", "")
            )
            mue_edits.append(mue_edit)
            if exceeds:
                mai_description = {
                    "1": "Line-level adjudication",
                    "2": "Claim-level adjudication",
                    "3": "Date of service adjudication"
                }
                errors.append({
                    "type": "mue_exceeded",
                    "message": f"MUE exceeded for {hcpcs}: {uom} units billed, limit is {mue_value}",
                    "code": hcpcs,
                    "units_billed": uom,
                    "mue_limit": mue_value,
                    "mai": mai,
                    "adjudication": mai_description.get(mai, "Unknown")
                })
        else:
            warnings.append({
                "type": "mue_not_found",
                "message": f"No MUE data found for {hcpcs}",
                "code": hcpcs
            })
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            ptp_edits=[],
            mue_edits=mue_edits,
            date_of_service=dos
        )

    def validate_claim(self, claim_data: Dict[str, Any]) -> ValidationResult:
        dos = claim_data["dos"]
        line_items = claim_data["line_items"]
        all_codes = [item["hcpcs"] for item in line_items]
        ptp_result = self.ptpcheck(all_codes, dos)
        all_mue_edits = []
        mue_errors = []
        mue_warnings = []
        for item in line_items:
            mue_result = self.muecheck(item["hcpcs"], int(item.get("units", 0) or 0), dos)
            # annotate errors with line sequence for better traceability
            for e in mue_result.errors:
                e["line_seq"] = item.get("line_seq")
            for w in mue_result.warnings:
                w["line_seq"] = item.get("line_seq")
            all_mue_edits.extend(mue_result.mue_edits)
            mue_errors.extend(mue_result.errors)
            mue_warnings.extend(mue_result.warnings)
        return ValidationResult(
            valid=(len(ptp_result.errors) == 0 and len(mue_errors) == 0),
            errors=ptp_result.errors + mue_errors,
            warnings=ptp_result.warnings + mue_warnings,
            ptp_edits=ptp_result.ptp_edits,
            mue_edits=all_mue_edits,
            date_of_service=dos
        )
# --- End: NCCIAgent code ---

# ---------- Helper functions for SQLite ingestion & date parsing ----------

def parse_date_to_yyyymmdd(val: Optional[str]) -> Optional[str]:
    if not val or not str(val).strip():
        return None
    s = str(val).strip()
    # Try common formats
    fmts = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d", "%Y/%m/%d", "%d-%b-%Y",
        "%b %d %Y", "%d-%m-%Y"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f).strftime("%Y%m%d")
        except Exception:
            continue
    # If it looks like YYYYMMDD already (digits only, len 8)
    if s.isdigit() and len(s) == 8:
        return s
    return None


def load_claims_from_sqlite(db_path: str) -> List[Dict[str, Any]]:
    q = (
        "SELECT Claim_ID, Line_Sequence_No, Procedure_Code_ID, Line_Units, "
        "Line_Service_Start_Date, Claim_Service_Start_Date, "
        "Modifier_Code1, Modifier_Code2, Modifier_Code3, Modifier_Code4 "
        "FROM ClaimsData "
        "WHERE Procedure_Code_ID IS NOT NULL AND TRIM(Procedure_Code_ID) != ''"
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(q).fetchall()
    finally:
        conn.close()

    claims: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        claim_id = str(r["Claim_ID"]).strip()
        line_seq = r["Line_Sequence_No"]
        hcpcs = str(r["Procedure_Code_ID"]).strip()
        units = r["Line_Units"] if r["Line_Units"] is not None else 0
        line_dos = parse_date_to_yyyymmdd(r["Line_Service_Start_Date"]) or parse_date_to_yyyymmdd(r["Claim_Service_Start_Date"]) or ""
        modifiers = [
            str(r["Modifier_Code1"]).strip() if r["Modifier_Code1"] else "",
            str(r["Modifier_Code2"]).strip() if r["Modifier_Code2"] else "",
            str(r["Modifier_Code3"]).strip() if r["Modifier_Code3"] else "",
            str(r["Modifier_Code4"]).strip() if r["Modifier_Code4"] else "",
        ]
        if not claim_id:
            # Skip if no claim id
            continue
        if claim_id not in claims:
            claims[claim_id] = {
                "claim_id": claim_id,
                "line_items": [],
                "all_line_dos": []
            }
        claims[claim_id]["line_items"].append({
            "hcpcs": hcpcs,
            "units": int(units or 0),
            "line_seq": line_seq,
            "dos": line_dos,
            "modifiers": [m for m in modifiers if m]
        })
        if line_dos:
            claims[claim_id]["all_line_dos"].append(line_dos)

    # finalize and set claim-level DOS as earliest line DOS (fallback to today if missing)
    finalized: List[Dict[str, Any]] = []
    today = datetime.today().strftime("%Y%m%d")
    for c in claims.values():
        if c["all_line_dos"]:
            claim_dos = min([d for d in c["all_line_dos"] if d])
        else:
            claim_dos = today
        finalized.append({
            "claim_id": c["claim_id"],
            "dos": claim_dos,
            "line_items": c["line_items"]
        })
    return finalized

# ---------- Aggregated MUE checks (MAI = 2 / 3) ----------

def aggregated_mue_checks(agent: NCCIAgent, claim: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return additional MUE errors for MAI=2 (claim) and MAI=3 (date of service)."""
    errs: List[Dict[str, Any]] = []
    # Build groups
    # MAI=2 -> sum by HCPCS across claim
    # MAI=3 -> sum by (HCPCS, DOS) across claim
    by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_code_dos: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for li in claim["line_items"]:
        code = li["hcpcs"]
        dos = li.get("dos") or claim["dos"]
        by_code[code].append(li)
        by_code_dos[(code, dos)].append(li)

    mai_desc = {"1": "Line-level adjudication", "2": "Claim-level adjudication", "3": "Date of service adjudication"}

    # MAI=2
    for code, lines in by_code.items():
        if code not in agent.mue_data:
            continue
        m = agent.mue_data[code]
        if str(m.get("mai")) != "2":
            continue
        total_units = sum(int(li.get("units", 0) or 0) for li in lines)
        limit = int(m.get("mue", 0))
        if total_units > limit:
            errs.append({
                "type": "mue_exceeded_aggregated",
                "scope": "claim",
                "message": f"MUE exceeded (MAI=2, claim-level) for {code}: total {total_units} units across claim, limit {limit}",
                "code": code,
                "total_units": total_units,
                "mue_limit": limit,
                "mai": "2",
                "adjudication": mai_desc.get("2"),
                "line_seqs": [li.get("line_seq") for li in lines]
            })

    # MAI=3
    for (code, dos), lines in by_code_dos.items():
        if code not in agent.mue_data:
            continue
        m = agent.mue_data[code]
        if str(m.get("mai")) != "3":
            continue
        total_units = sum(int(li.get("units", 0) or 0) for li in lines)
        limit = int(m.get("mue", 0))
        if total_units > limit:
            errs.append({
                "type": "mue_exceeded_aggregated",
                "scope": "date_of_service",
                "message": f"MUE exceeded (MAI=3, DOS) for {code} on {dos}: total {total_units} units, limit {limit}",
                "code": code,
                "dos": dos,
                "total_units": total_units,
                "mue_limit": limit,
                "mai": "3",
                "adjudication": mai_desc.get("3"),
                "line_seqs": [li.get("line_seq") for li in lines]
            })

    return errs

# ---------- CSV writer ----------

def write_flags_csv(out_path: str, claim_id: str, claim_dos: str, claim: Dict[str, Any],
                    validation: ValidationResult, extra_mue_errs: List[Dict[str, Any]]):
    fieldnames = [
        "Claim_ID", "Flag_Level", "Line_Sequence_Nos", "HCPCS_Codes", "Flag_Type",
        "Severity", "Reason", "Modifier_Indicator", "MAI", "DOS", "Units_Billed", "MUE_Limit"
    ]
    file_exists = Path(out_path).exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()

        # --- PTP errors/warnings ---
        # Map codes -> line sequences for traceability
        codes_to_lines: Dict[str, List[Any]] = defaultdict(list)
        for li in claim["line_items"]:
            codes_to_lines[li["hcpcs"]].append(li.get("line_seq"))

        for severity, items in (("ERROR", validation.errors), ("WARNING", validation.warnings)):
            for it in items:
                if it.get("type") in ("unbundling", "mutually_exclusive"):
                    code_a, code_b = it.get("codes", [None, None])
                    line_pairs = []
                    # produce combinations of involved line seqs
                    for la in codes_to_lines.get(code_a, [None]):
                        for lb in codes_to_lines.get(code_b, [None]):
                            if la is not None and lb is not None:
                                line_pairs.append(f"{la}|{lb}")
                    w.writerow({
                        "Claim_ID": claim_id,
                        "Flag_Level": "CLAIM",
                        "Line_Sequence_Nos": ";".join(line_pairs) if line_pairs else "",
                        "HCPCS_Codes": f"{code_a},{code_b}",
                        "Flag_Type": "PTP",
                        "Severity": severity,
                        "Reason": it.get("message"),
                        "Modifier_Indicator": it.get("modifier_indicator"),
                        "MAI": "",
                        "DOS": claim_dos,
                        "Units_Billed": "",
                        "MUE_Limit": ""
                    })
                elif it.get("type") in ("mue_exceeded",):
                    # This is line-level mue from validate_claim
                    code = it.get("code")
                    line_seq = it.get("line_seq")
                    w.writerow({
                        "Claim_ID": claim_id,
                        "Flag_Level": "LINE",
                        "Line_Sequence_Nos": str(line_seq) if line_seq is not None else ";".join(map(str, codes_to_lines.get(code, []))),
                        "HCPCS_Codes": code,
                        "Flag_Type": "MUE",
                        "Severity": severity,
                        "Reason": it.get("message"),
                        "Modifier_Indicator": "",
                        "MAI": it.get("mai"),
                        "DOS": claim_dos,
                        "Units_Billed": it.get("units_billed"),
                        "MUE_Limit": it.get("mue_limit")
                    })
                elif it.get("type") == "mue_not_found":
                    code = it.get("code")
                    line_seq = it.get("line_seq")
                    w.writerow({
                        "Claim_ID": claim_id,
                        "Flag_Level": "LINE",
                        "Line_Sequence_Nos": str(line_seq) if line_seq is not None else ";".join(map(str, codes_to_lines.get(code, []))),
                        "HCPCS_Codes": code,
                        "Flag_Type": "MUE",
                        "Severity": severity,
                        "Reason": it.get("message"),
                        "Modifier_Indicator": "",
                        "MAI": "",
                        "DOS": claim_dos,
                        "Units_Billed": "",
                        "MUE_Limit": ""
                    })

        # --- Aggregated MUE errors (MAI 2/3) ---
        for it in extra_mue_errs:
            code = it.get("code")
            line_seqs = it.get("line_seqs", [])
            dos_val = it.get("dos", claim_dos)
            w.writerow({
                "Claim_ID": claim_id,
                "Flag_Level": "CLAIM" if it.get("scope") == "claim" else "DOS",
                "Line_Sequence_Nos": ";".join(map(str, line_seqs)),
                "HCPCS_Codes": code,
                "Flag_Type": "MUE",
                "Severity": "ERROR",
                "Reason": it.get("message"),
                "Modifier_Indicator": "",
                "MAI": it.get("mai"),
                "DOS": dos_val,
                "Units_Billed": it.get("total_units"),
                "MUE_Limit": it.get("mue_limit")
            })

# ---------- Main entry ----------

def main():
    ap = argparse.ArgumentParser(description="Run NCCI validation against claims in SQLite and export CSV flags")
    ap.add_argument("--db", required=True, help="Path to SQLite database (e.g., claims_database.db)")
    ap.add_argument("--out", default="ncci_validation_results.csv", help="Output CSV path")
    ap.add_argument("--auto-fetch", action="store_true", help="Fetch latest NCCI data from CMS (internet required)")
    ap.add_argument("--year", type=int, help="Year for NCCI file fetch (optional)")
    ap.add_argument("--quarter", type=int, choices=[1,2,3,4], help="Quarter for NCCI file fetch (optional)")
    args = ap.parse_args()

    config = {
        "name": "cms-ncci",
        "resources": [
            {"uri": "https://www.cms.gov/medicare/coding-billing/ncci-medicare", "description": "PTP rules & manuals"},
            {"uri": "https://www.cms.gov/medicare/coding-billing/ncci-medically-unlikely-edits", "description": "MUE tables"}
        ]
    }

    agent = NCCIAgent(config, auto_fetch=args.auto_fetch, year=args.year, quarter=args.quarter)

    claims = load_claims_from_sqlite(args.db)
    logger.info(f"Loaded {len(claims)} claims from SQLite")

    # Ensure output dir exists and clear old file if present
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    for claim in claims:
        claim_id = claim["claim_id"]
        claim_dos = claim["dos"]
        # validate
        validation = agent.validate_claim({
            "dos": claim_dos,
            "line_items": claim["line_items"],
        })
        # aggregated MUE for MAI 2/3
        extra_mue = aggregated_mue_checks(agent, claim)
        # write rows
        write_flags_csv(str(out_path), claim_id, claim_dos, claim, validation, extra_mue)

    logger.info(f"Completed. CSV written to: {out_path}")

if __name__ == "__main__":
    main()
