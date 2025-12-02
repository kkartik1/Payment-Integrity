import os
import pandas as pd

# -------------------------
# Config
# -------------------------
INPUT_FILE = os.path.join("Data", "prepared_claims_data.csv")
OUTPUT_FILE = os.path.join("Data", "biobert_upcoding_features_linelevel.csv")

LINE_SEP = " | "
FIELD_SEP = "; "

DIAGNOSIS_COLS = [
    "Primary_Diagnosis_Code_Description",
    "Diagnosis_Code_1_Description",
    "Diagnosis_Code_2_Description",
    "Diagnosis_Code_3_Description",
    "Diagnosis_Code_4_Description",
    "Diagnosis_Code_5_Description",
    "Diagnosis_Code_6_Description",
    "Diagnosis_Code_7_Description",
    "Diagnosis_Code_8_Description",
]

SECTION_SPECS = [
    ("Diagnosis", DIAGNOSIS_COLS, "; "),
    ("Procedure", ["Procedure_Description"], "; "),
    ("Modifiers", ["Modifier_Description1", "Modifier_Description2", "Modifier_Description3", "Modifier_Description4"], "; "),
    ("Place Of Service", ["Place_Of_Service_Indicator_Description"], "; "),
    ("Provider Taxonomy", ["Provider_Taxonomy_Description", "Claim_Network_Indicator"], ", "),
    ("Member", ["Member_Gender", "Member_Age"], ", "),
    ("Provider State", ["Provider_State"], "; "),
]

TEXTUAL_COLS = [
    "Primary_Diagnosis_Code_Description",
    "Diagnosis_Code_1_Description",
    "Diagnosis_Code_2_Description",
    "Diagnosis_Code_3_Description",
    "Diagnosis_Code_4_Description",
    "Diagnosis_Code_5_Description",
    "Diagnosis_Code_6_Description",
    "Diagnosis_Code_7_Description",
    "Diagnosis_Code_8_Description",
    "Member_Gender",
    "Member_Age",
    "Provider_State",
    "Procedure_Description",
    "Modifier_Description1",
    "Modifier_Description2",
    "Modifier_Description3",
    "Modifier_Description4",
    "Place_Of_Service_Indicator_Description",
    "Provider_Taxonomy_Description",
]

# -------------------------
# Load data
# -------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, dtype="string")

if "Claim_ID" not in df.columns:
    raise KeyError("Claim_ID column missing in input data")

available_textual_cols = [c for c in TEXTUAL_COLS if c in df.columns]
if not available_textual_cols:
    raise KeyError("None of the expected textual columns are present in the input data.")

section_specs = []
for label, cols, joiner in SECTION_SPECS:
    present = [c for c in cols if c in df.columns]
    if present:
        section_specs.append((label, present, joiner))

# -------------------------
# Helpers
# -------------------------
def _row_values(row, cols):
    vals = []
    for c in cols:
        v = row.get(c, None)
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            vals.append(s)
    return vals

def build_line_text(row):
    section_lines = []
    for label, cols, joiner in section_specs:
        vals = _row_values(row, cols)
        if vals:
            section_lines.append(f"{label}: " + joiner.join(vals))
    covered_cols = set()
    for _, cols, _ in section_specs:
        covered_cols.update(cols)
    other_cols = [c for c in available_textual_cols if c not in covered_cols]
    other_vals = _row_values(row, other_cols)
    if other_vals:
        section_lines.append("Other: " + FIELD_SEP.join(other_vals))
    return LINE_SEP.join(section_lines)

# -------------------------
# Build line-level features
# -------------------------
df = df.sort_values(["Claim_ID"]).reset_index(drop=True)
df["LineSeq"] = df.groupby("Claim_ID").cumcount() + 1
df["Claim_ID_LineSeq"] = df["Claim_ID"].astype(str) + "_" + df["LineSeq"].astype(str)
df["bio_text"] = df.apply(build_line_text, axis=1)

cols = ["Claim_ID_LineSeq", "Claim_ID", "bio_text"] + (["Label"] if "Label" in df.columns else [])
df_out = df[cols]

os.makedirs("Data", exist_ok=True)
df_out.to_csv(OUTPUT_FILE, index=False)
print(f"Line-level feature preparation complete. Saved {len(df_out)} lines to {OUTPUT_FILE}.")
print("Sample rows:")
print(df_out.head(5))