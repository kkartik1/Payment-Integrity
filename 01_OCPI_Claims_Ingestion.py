import pandas as pd
import sqlite3

# -----------------------------
# Step 1: Read Excel Sheets
# -----------------------------
master_file = "Master.xlsx"
mapping_file = "column_mapping.xlsx"

# Read Super and Flag tabs
super_df = pd.read_excel(master_file, sheet_name="Super")
flag_df = pd.read_excel(master_file, sheet_name="Flag")

# Read column mapping
mapping_df = pd.read_excel(mapping_file)

# -----------------------------
# Step 2: Merge DataFrames on CLCL_ID
# -----------------------------
combined_df = pd.merge(super_df, flag_df, on="CLCL_ID", how="left")

# Add additional columns
combined_df["Savings Status"] = combined_df.get("Savings Status", "")
combined_df["Denial_Savings"] = combined_df.get("Denial_Savings", "")

# -----------------------------
# Step 3: Rename Columns using mapping
# -----------------------------

rename_dict = dict(zip(mapping_df["Input File"], mapping_df["New Descriptive Name"]))
combined_df.rename(columns=rename_dict, inplace=True)

# -----------------------------
# Step 4: Prepare for SQLite
# -----------------------------
combined_df["Label"] = combined_df["Savings Status"]
combined_df["Label_Type"] = combined_df["Denial_Savings"]

print(combined_df.columns)
# -----------------------------
# Step 5: Create SQLite DB and Table
# -----------------------------
db_name = "claims_database.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()
cursor.execute("""
DROP TABLE IF EXISTS ClaimsData;
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS ClaimsData (
 Diagnosis_Code_1 TEXT,
 Diagnosis_Code_2 TEXT,
 Diagnosis_Code_3 TEXT,
 Diagnosis_Code_4 TEXT,
 Diagnosis_Code_5 TEXT,
 Diagnosis_Code_6 TEXT,
 Diagnosis_Code_7 TEXT,
 Diagnosis_Code_8 TEXT,
 Primary_Diagnosis_Code TEXT,
 Claim_ID TEXT,
 Line_Sequence_No INTEGER,
 Claim_Sub_Type TEXT,
 Claim_Total_Charge REAL,
 Claim_Total_Payable REAL,
 Claim_Paid_Date TEXT,
 Claim_Service_Start_Date TEXT,
 Claim_Service_End_Date TEXT,
 Claim_Network_Indicator TEXT,
 Claim_Patient_Account_No TEXT,
 Line_Current_Status TEXT,
 Line_Charge_Amount REAL,
 Line_Paid_Amount REAL,
 Line_Allowed_Amount REAL,
 Line_Units INTEGER,
 Line_Service_Start_Date TEXT,
 Line_Service_End_Date TEXT,
 Place_Of_Service_Indicator TEXT,
 Professional_Component_Indicator TEXT,
 Primary_Payment_Amount REAL,
 Coinsurance_Amount REAL,
 Copay_Amount REAL,
 Deductible_Amount REAL,
 Disallowed_Amount REAL,
 Disallowed_Exception_Code TEXT,
 Discount_Amount REAL,
 Risk_Withhold_Amount REAL,
 Allowed_Units INTEGER,
 Member_ID TEXT,
 Member_Gender TEXT,
 Member_Date_Of_Birth TEXT,
 Member_Relationship TEXT,
 Provider_ID TEXT,
 Service_Provider_Name TEXT,
 Service_Provider_City TEXT,
 Service_Provider_State TEXT,
 Service_Provider_Zip TEXT,
 Provider_NPI TEXT,
 Provider_Taxonomy_Code TEXT,
 Group_Name TEXT,
 Group_ID TEXT,
 Admission_Date TEXT,
 Discharge_Date TEXT,
 Bill_Type TEXT,
 Procedure_Code_ID TEXT,
 Modifier_Code1 TEXT,
 Modifier_Code2 TEXT,
 Modifier_Code3 TEXT,
 Modifier_Code4 TEXT,
 Procedure_Description TEXT,
 Authorization_Description TEXT,
 Label TEXT,
 Label_Type TEXT
);
""")

# -----------------------------
# Step 6: Insert Data
# -----------------------------
table_columns = [col for col in combined_df.columns if col in [
    "Diagnosis_Code_1","Diagnosis_Code_2","Diagnosis_Code_3","Diagnosis_Code_4","Diagnosis_Code_5","Diagnosis_Code_6","Diagnosis_Code_7","Diagnosis_Code_8",
    "Primary_Diagnosis_Code","Claim_ID","Line_Sequence_No","Claim_Sub_Type","Claim_Total_Charge","Claim_Total_Payable","Claim_Paid_Date","Claim_Service_Start_Date","Claim_Service_End_Date",
    "Claim_Network_Indicator","Claim_Patient_Account_No","Line_Current_Status","Line_Charge_Amount","Line_Paid_Amount","Line_Allowed_Amount","Line_Units","Line_Service_Start_Date","Line_Service_End_Date",
    "Place_Of_Service_Indicator","Professional_Component_Indicator","Primary_Payment_Amount","Coinsurance_Amount","Copay_Amount","Deductible_Amount","Disallowed_Amount","Disallowed_Exception_Code",
    "Discount_Amount","Risk_Withhold_Amount","Allowed_Units","Member_ID","Member_Gender","Member_Date_Of_Birth","Member_Relationship","Provider_ID","Service_Provider_Name","Service_Provider_City",
    "Service_Provider_State","Service_Provider_Zip","Provider_NPI","Provider_Taxonomy_Code","Group_Name","Group_ID","Admission_Date","Discharge_Date","Bill_Type","Procedure_Code_ID","Modifier_Code1",
    "Modifier_Code2", "Modifier_Code3", "Modifier_Code4","Procedure_Description", "Authorization_Description","Label","Label_Type"
]]

insert_df = combined_df[table_columns]
insert_df.to_sql("ClaimsData", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print("Data successfully inserted into claims_database.db")