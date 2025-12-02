"""
Claims Data Preparation (v3)

- Reads from SQLite table: ClaimsData
- Converts Label -> binary (1 if 'Y' else 0)
- Enriches with descriptions for:
  * Primary_Diagnosis_Code + Diagnosis_Code_1..8 -> icd_codes.description
  * Place_Of_Service_Indicator                -> pos_codes.description
  * Provider_Taxonomy_Code                    -> provider_taxonomy.display_name
  * Modifier_Code1..Modifier_Code4            -> modifier_codes.short_description (if table exists)
- Does NOT fetch/append descriptions for DRG, revenue code, or procedure codes
- Writes CSV to Data/prepared_claims_data.csv
"""

import os
import re
from datetime import datetime
from num2words import num2words
import logging
import pandas as pd
from sqlalchemy import create_engine, text

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("claims_prep_v3")

us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "Virgin Islands, U.S.": "VI",
}

# -------------------------
# Helpers
# -------------------------
def normalize_code(val) -> str:
    """
    Normalize a medical code for lookups:
    - Convert to string
    - Uppercase
    - Remove non-alphanumerics (incl. dots, dashes, spaces)
    """
    if pd.isna(val):
        return ""
    s = str(val).strip().upper()
    # Keep only A-Z, 0-9
    return re.sub(r"[^A-Z0-9]", "", s)

def build_normalized_mapping(df: pd.DataFrame, key_col: str, val_col: str) -> dict:
    """
    Build a dict with normalized keys from mapping table.
    """
    if df.empty:
        return {}
    keys = df[key_col].astype(str).map(normalize_code)
    vals = df[val_col]
    return dict(zip(keys, vals))

def has_table(engine, table_name: str) -> bool:
    """
    Check if a table exists in the connected SQLite DB.
    """
    query = text(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = :name LIMIT 1;"
    )
    with engine.connect() as conn:
        res = conn.execute(query, {"name": table_name}).scalar()
    return bool(res)

# -------------------------
# Main class
# -------------------------
class ClaimsDataPreparationV3:
    def __init__(self, db_path: str = "claims_database.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        logger.info(f"Using database: {self.db_path}")

        # Mapping caches
        self.icd_map = {}
        self.pos_map = {}
        self.taxonomy_map = {}
        self.modifier_map = {}

    # -------- Load base data --------
    def load_claims(self) -> pd.DataFrame:
        """
        Load ALL columns from ClaimsData (only use what's available).
        """
        query = "SELECT * FROM ClaimsData;"
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Loaded {len(df):,} rows from ClaimsData")
        return df

    # -------- Load lookup tables --------
    def load_icd_mapping(self):
        if not has_table(self.engine, "icd_codes"):
            logger.warning("Table 'icd_codes' not found; diagnosis descriptions will be empty.")
            self.icd_map = {}
            return
        df = pd.read_sql_query("SELECT icd_code, description FROM icd_codes;", self.engine)
        self.icd_map = build_normalized_mapping(df, "icd_code", "description")
        logger.info(f"ICD mapping loaded: {len(self.icd_map):,} codes")

    def load_pos_mapping(self):
        if not has_table(self.engine, "pos_codes"):
            logger.warning("Table 'pos_codes' not found; POS descriptions will be empty.")
            self.pos_map = {}
            return
        df = pd.read_sql_query("SELECT pos_code, description FROM pos_codes;", self.engine)
        self.pos_map = build_normalized_mapping(df, "pos_code", "description")
        logger.info(f"POS mapping loaded: {len(self.pos_map):,} codes")

    def load_taxonomy_mapping(self):
        if not has_table(self.engine, "provider_taxonomy"):
            logger.warning("Table 'provider_taxonomy' not found; taxonomy descriptions will be empty.")
            self.taxonomy_map = {}
            return
        # Use display_name for taxonomy description (as requested)
        df = pd.read_sql_query("SELECT code, display_name FROM provider_taxonomy;", self.engine)
        self.taxonomy_map = build_normalized_mapping(df, "code", "display_name")
        logger.info(f"Provider taxonomy mapping loaded: {len(self.taxonomy_map):,} codes")

    def load_modifier_mapping(self):
        # Optional: many environments may not have this table
        if not has_table(self.engine, "modifier_codes"):
            logger.warning(
                "Table 'modifier_codes' not found; Modifier_Code1..4 descriptions will be empty."
            )
            self.modifier_map = {}
            return
        df = pd.read_sql_query("SELECT modifier, short_description FROM modifier_codes;", self.engine)
        self.modifier_map = build_normalized_mapping(df, "modifier", "short_description")
        logger.info(f"Modifier mapping loaded: {len(self.modifier_map):,} codes")

    # -------- Enrichment --------
    def age_to_ordinal_string(self, age):
        """Converts a numerical age to an ordinal string format (e.g., '30th years')"""
        # num2words expects a single integer as input
        return num2words(age, 'ordinal') + ' years'

    def add_label_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update existing Label column: 1 if 'Y', else 0.
        If Label is missing, creates column with 0s.
        """
        if "Label" not in df.columns:
            logger.warning("Column 'Label' not found; creating with default 0s.")
            df["Label"] = 0
            return df

        df["Label"] = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("Y")
            .astype(int)
        )
        
        gender_mapping = {
            'M': 'Male',
            'F': 'Female',
            ' ': 'Unknown'
        }
        df['Member_Gender'] = df['Member_Gender'].replace(gender_mapping)
        
        network_mapping = {
            'I': 'In Network',
            'O': 'Out of Network'
        }
        df['Claim_Network_Indicator'] = df['Claim_Network_Indicator'].replace(network_mapping)
        df.loc[~df['Claim_Network_Indicator'].isin(['In Network', 'Out of Network']), 'Claim_Network_Indicator'] = 'Unknown'
        
        df['Member_Date_Of_Birth'] = pd.to_datetime(df['Member_Date_Of_Birth'])
        # Calculate the ages as a Series first
        ages_numeric = datetime.now().year - df['Member_Date_Of_Birth'].dt.year
        # Apply the custom function row-by-row to the Series
        df['Member_Age'] = ages_numeric.apply(self.age_to_ordinal_string)
        
        us_state_to_abbrev = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY",
            "District of Columbia": "DC",
            "American Samoa": "AS",
            "Guam": "GU",
            "Northern Mariana Islands": "MP",
            "Puerto Rico": "PR",
            "United States Minor Outlying Islands": "UM",
            "Virgin Islands, U.S.": "VI",
        }
    
        # invert the dictionary
        abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))
        df['Provider_State'] = df['Service_Provider_State'].apply(lambda x: abbrev_to_us_state.get(x))
        return df

    def _map_series(self, s: pd.Series, mapping: dict) -> pd.Series:
        # Return None for blanks/NaNs; otherwise lookup normalized key
        return s.map(lambda x: mapping.get(normalize_code(x), None) if pd.notna(x) and str(x).strip() != "" else None)

    def add_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add description columns for:
        - Primary_Diagnosis_Code, Diagnosis_Code_1..8
        - Place_Of_Service_Indicator
        - Provider_Taxonomy_Code
        - Modifier_Code1..Modifier_Code4
        """
        # ICD descriptions
        icd_cols = []
        if "Primary_Diagnosis_Code" in df.columns:
            icd_cols.append("Primary_Diagnosis_Code")
        for i in range(1, 9):
            col = f"Diagnosis_Code_{i}"
            if col in df.columns:
                icd_cols.append(col)
        for col in icd_cols:
            out_col = f"{col}_Description"
            df[out_col] = self._map_series(df[col], self.icd_map)

        # POS description
        if "Place_Of_Service_Indicator" in df.columns:
            df["Place_Of_Service_Indicator_Description"] = self._map_series(
                df["Place_Of_Service_Indicator"], self.pos_map
            )

        # Provider taxonomy description (use display_name)
        if "Provider_Taxonomy_Code" in df.columns:
            df["Provider_Taxonomy_Description"] = self._map_series(
                df["Provider_Taxonomy_Code"], self.taxonomy_map
            )

        # Modifier descriptions (1..4 if present)
        for i in range(1, 5):
            col = f"Modifier_Code{i}"
            if col in df.columns:
                out_col = f"Modifier_Description{i}"
                df[out_col] = self._map_series(df[col], self.modifier_map)

        # NOTE: DRG, revenue codes, and procedure codes intentionally ignored.
        return df

    # -------- Orchestration --------
    def run(self) -> pd.DataFrame:
        # Load base data
        claims = self.load_claims()
        # Binary Label
        claims = self.add_label_binary(claims)
        # Load mappings we actually need
        self.load_icd_mapping()
        self.load_pos_mapping()
        self.load_taxonomy_mapping()
        self.load_modifier_mapping()
        # Add descriptions
        claims = self.add_descriptions(claims)
        return claims

    def save(self, df: pd.DataFrame, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved prepared data to: {out_path} (rows: {len(df):,})")


def main():
    prep = ClaimsDataPreparationV3()
    prepared = prep.run()
    prep.save(prepared, os.path.join("Data", "prepared_claims_data.csv"))


if __name__ == "__main__":
    main()
