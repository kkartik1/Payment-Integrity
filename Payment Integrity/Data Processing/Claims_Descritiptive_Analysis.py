
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


class HealthcareClaimsAnalyzer:
    """
    Descriptive analysis for healthcare claims data (SQLite)
    adapted to the schema:
      - Database: claims_database.db
      - Table:    ClaimsData
      - Upcoding label field: Label (boolean-ish/indicator)
    """

    def __init__(self, db_path: str = 'claims_database.db', output_dir: str = 'analysis_output'):
        """Initialize with database connection and output directory"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.df = None  # line-level working dataframe
        self.output_dir = output_dir

        # Create output directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(f"{self.output_dir}/visualizations").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/csv_outputs").mkdir(exist_ok=True)
        print(f"Output directory created: {self.output_dir}")

    # -----------------------
    # Data loading & wrangling
    # -----------------------
    def _standardize_label(self, s: pd.Series) -> pd.Series:
        """Convert Label field into boolean upcoding indicator.
        Accepts common truthy/falsey forms (1/0, yes/no, y/n, true/false, upcoded/clean).
        """
        if s is None:
            return pd.Series(dtype=bool)
        def to_bool(x):
            if pd.isna(x):
                return False
            if isinstance(x, (int, float)):
                return bool(int(x))
            x_str = str(x).strip().lower()
            return x_str in {'1', 'y', 'yes', 'true', 't', 'upcoded', 'fraud', 'suspicious'}
        return s.apply(to_bool)

    def load_data(self, table_name: str = 'ClaimsData') -> pd.DataFrame:
        """Load data from SQLite database and align to expected working columns"""
        query = f"SELECT * FROM {table_name}"
        self.df = pd.read_sql_query(query, self.conn)

        # --- Column mapping from provided schema to standardized working names
        rename_map = {
            # identifiers
            'Claim_ID': 'claim_id',
            'Line_Sequence_No': 'line_number',
            'Member_ID': 'member_id',
            'Provider_ID': 'provider_id',
            'Provider_NPI': 'provider_npi',
            'Provider_Taxonomy_Code': 'provider_taxonomy',
            'Group_ID': 'group_id',
            'Group_Name': 'group_name',
            # dates
            'Claim_Service_Start_Date': 'service_date_header',
            'Claim_Service_End_Date': 'service_end_date_header',
            'Line_Service_Start_Date': 'service_date_line',
            'Line_Service_End_Date': 'service_end_date_line',
            'Admission_Date': 'admission_date',
            'Discharge_Date': 'discharge_date',
            'Claim_Paid_Date': 'paid_date',
            'Member_Date_Of_Birth': 'member_dob',
            # diagnosis
            'Primary_Diagnosis_Code': 'primary_diagnosis',
            # financials (header-level)
            'Claim_Total_Charge': 'total_charges_header',
            'Claim_Total_Payable': 'expected_payment_header',
            # financials (line-level)
            'Line_Charge_Amount': 'total_charges_line',
            'Line_Paid_Amount': 'paid_amount_line',
            'Line_Allowed_Amount': 'allowed_amount_line',
            'Line_Units': 'units',
            'Allowed_Units': 'allowed_units',
            # other attributes
            'Place_Of_Service_Indicator': 'place_of_service',
            'Claim_Sub_Type': 'claim_type',
            'Bill_Type': 'bill_type',
            'Claim_Network_Indicator': 'network_indicator',
            'Professional_Component_Indicator': 'professional_component',
            'Member_Gender': 'member_gender',
            'Member_Relationship': 'member_relationship',
            'Service_Provider_Name': 'provider_name',
            'Service_Provider_City': 'provider_city',
            'Service_Provider_State': 'provider_state',
            'Service_Provider_Zip': 'provider_zip',
            # procedure & modifiers
            'Procedure_Code_ID': 'procedure_code',
            'Modifier_Code1': 'modifier1',
            'Modifier_Code2': 'modifier2',
            'Modifier_Code3': 'modifier3',
            'Modifier_Code4': 'modifier4',
            'Procedure_Description': 'procedure_description',
            # upcoding labels
            'Label': 'label',
            'Label_Type': 'label_type',
            # misc
            'Authorization_Description': 'authorization_description',
        }
        self.df.rename(columns=rename_map, inplace=True)

        # Derive secondary diagnosis columns list (Diagnosis_Code_1..8)
        self.secondary_dx_cols = [c for c in self.df.columns if c.startswith('Diagnosis_Code_')]

        # Convert date columns
        date_cols = [
            'service_date_header', 'service_end_date_header',
            'service_date_line', 'service_end_date_line',
            'admission_date', 'discharge_date', 'paid_date', 'member_dob'
        ]
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Length of stay (in days) when admission/discharge available
        if 'admission_date' in self.df.columns and 'discharge_date' in self.df.columns:
            self.df['length_of_stay'] = (self.df['discharge_date'] - self.df['admission_date']).dt.days

        # Normalize label to boolean
        if 'label' in self.df.columns:
            self.df['is_upcoded_header'] = self._standardize_label(self.df['label'])
        else:
            self.df['is_upcoded_header'] = False

        # If desired, mirror label as line-level flag for plotting consistency
        self.df['is_upcoded_line'] = self.df['is_upcoded_header']

        # Build helper columns
        # Unit cost at line
        if {'total_charges_line', 'units'}.issubset(self.df.columns):
            self.df['unit_cost'] = self.df['total_charges_line'] / self.df['units'].replace(0, np.nan)

        # Year-month for temporal grouping (from service_date_header if available else paid_date)
        base_date = 'service_date_header' if 'service_date_header' in self.df.columns else 'paid_date'
        if base_date in self.df.columns:
            self.df['year_month'] = self.df[base_date].dt.to_period('M')

        # Ensure key identifier columns exist to avoid KeyErrors later
        for must_have, fallback in [
            ('claim_id', None),
            ('member_id', None),
            ('provider_id', None),
        ]:
            if must_have not in self.df.columns:
                self.df[must_have] = fallback

        print(f"Data loaded successfully: {len(self.df):,} records")
        return self.df

    # -----------------------
    # Basic statistics
    # -----------------------
    def basic_statistics(self):
        print("\n" + "="*80)
        print("1. BASIC DATA STATISTICS")
        print("="*80)
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Missing values analysis
        print("\n--- Missing Values Analysis ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        print(missing_df[missing_df['Missing_Count'] > 0].head(25))

        # Data types
        print("\n--- Data Types ---")
        print(self.df.dtypes.value_counts())
        return missing_df

    # -----------------------
    # Claims overview
    # -----------------------
    def claims_overview(self):
        print("\n" + "="*80)
        print("2. CLAIMS OVERVIEW")
        print("="*80)
        # Unique counts
        print("\n--- Unique Entity Counts ---")
        entities = {
            'Unique Claims': self.df['claim_id'].nunique() if 'claim_id' in self.df else np.nan,
            'Unique Members': self.df['member_id'].nunique() if 'member_id' in self.df else np.nan,
            'Unique Providers': self.df['provider_id'].nunique() if 'provider_id' in self.df else np.nan,
            'Unique Provider NPIs': self.df['provider_npi'].nunique() if 'provider_npi' in self.df else np.nan
        }
        for k, v in entities.items():
            print(f"{k}: {v:,}" if pd.notna(v) else f"{k}: N/A")

        # Claims per member
        if {'member_id', 'claim_id'}.issubset(self.df.columns):
            claims_per_member = self.df.groupby('member_id')['claim_id'].nunique()
            print(f"\n--- Claims Distribution per Member ---")
            print(f"Mean: {claims_per_member.mean():.2f}")
            print(f"Median: {claims_per_member.median():.2f}")
            print(f"Max: {claims_per_member.max()}")
            print(f"Members with >10 claims: {(claims_per_member > 10).sum():,}")
        else:
            claims_per_member = pd.Series(dtype=float)

        # Line items per claim
        if {'claim_id', 'line_number'}.issubset(self.df.columns):
            lines_per_claim = self.df.groupby('claim_id')['line_number'].count()
            print(f"\n--- Line Items per Claim ---")
            print(f"Mean: {lines_per_claim.mean():.2f}")
            print(f"Median: {lines_per_claim.median():.2f}")
            print(f"Max: {lines_per_claim.max()}")
        else:
            lines_per_claim = pd.Series(dtype=float)

        return entities, claims_per_member, lines_per_claim

    # -----------------------
    # Financial analysis
    # -----------------------
    def financial_analysis(self):
        print("\n" + "="*80)
        print("3. FINANCIAL ANALYSIS")
        print("="*80)

        # Header-level charges
        if 'total_charges_header' in self.df.columns:
            print("\n--- Total Charges (Header Level) ---")
            print(self.df['total_charges_header'].describe())
            print(f"Total Charges Sum: ${self.df['total_charges_header'].sum():,.2f}")

        # Line-level charges
        if 'total_charges_line' in self.df.columns:
            print("\n--- Total Charges (Line Level) ---")
            print(self.df['total_charges_line'].describe())
            print(f"Total Charges Sum: ${self.df['total_charges_line'].sum():,.2f}")

        # Payment vs allowed amount (line-level if available)
        if {'expected_payment_header'}.issubset(self.df.columns):
            print("\n--- Payment Analysis ---")
            print(f"Total Expected Payment (header): ${self.df['expected_payment_header'].sum():,.2f}")
        if 'allowed_amount_line' in self.df.columns:
            print(f"Total Allowed Amount (line, sum of rows): ${self.df['allowed_amount_line'].sum():,.2f}")

        # Cost per unit (line level)
        if {'total_charges_line', 'units'}.issubset(self.df.columns):
            self.df['cost_per_unit'] = self.df['total_charges_line'] / self.df['units'].replace(0, np.nan)
            print(f"\n--- Cost per Unit (line) ---")
            print(self.df['cost_per_unit'].describe())

        # Charge-to-payment ratio (header)
        if {'total_charges_header', 'expected_payment_header'}.issubset(self.df.columns):
            self.df['charge_to_payment_ratio'] = (
                self.df['total_charges_header'] / self.df['expected_payment_header'].replace(0, np.nan)
            )
            print(f"\n--- Charge-to-Payment Ratio (Header) ---")
            print(self.df['charge_to_payment_ratio'].describe())

        summary_cols = [c for c in ['total_charges_header', 'expected_payment_header'] if c in self.df.columns]
        return self.df[summary_cols].describe() if summary_cols else None

    # -----------------------
    # Upcoding analysis
    # -----------------------
    def upcoding_analysis(self):
        print("\n" + "="*80)
        print("4. UPCODING ANALYSIS (CRITICAL)")
        print("="*80)

        if 'is_upcoded_header' not in self.df.columns:
            print("Label field not found; skipping upcoding analysis.")
            return {}

        upcoded_header = self.df['is_upcoded_header'].sum()
        upcoded_header_pct = (upcoded_header / len(self.df) * 100) if len(self.df) else 0
        print(f"\n--- Header-Level Upcoding ---")
        print(f"Upcoded Rows: {upcoded_header:,} ({upcoded_header_pct:.2f}%)")
        print(f"Clean Rows: {(~self.df['is_upcoded_header']).sum():,}")

        # Upcoding types (from Label_Type if available)
        if 'label_type' in self.df.columns:
            print(f"\n--- Upcoding Types (Header / Label_Type) ---")
            print(self.df['label_type'].value_counts(dropna=False).head(20))

        # Financial impact
        if 'total_charges_header' in self.df.columns:
            upcoded_mask = self.df['is_upcoded_header'] == True
            clean_mask = self.df['is_upcoded_header'] == False
            upcoded_charges = self.df.loc[upcoded_mask, 'total_charges_header'].sum()
            clean_charges = self.df.loc[clean_mask, 'total_charges_header'].sum()
            print(f"\n--- Financial Impact ---")
            print(f"Charges from Upcoded Rows: ${upcoded_charges:,.2f}")
            print(f"Charges from Clean Rows:   ${clean_charges:,.2f}")
            if upcoded_mask.any():
                print(f"Avg Charge (Upcoded): ${self.df.loc[upcoded_mask, 'total_charges_header'].mean():,.2f}")
            if clean_mask.any():
                print(f"Avg Charge (Clean):   ${self.df.loc[clean_mask, 'total_charges_header'].mean():,.2f}")

        return {'upcoded_header_pct': upcoded_header_pct}

    # -----------------------
    # Provider analysis
    # -----------------------
    def provider_analysis(self):
        print("\n" + "="*80)
        print("5. PROVIDER ANALYSIS")
        print("="*80)

        specialty_col = 'provider_taxonomy' if 'provider_taxonomy' in self.df.columns else None
        if specialty_col:
            print("\n--- Provider Taxonomy Distribution (Top 15) ---")
            print(self.df[specialty_col].value_counts().head(15))

            print("\n--- Upcoding Rate by Taxonomy (Top 10) ---")
            specialty_upcode = self.df.groupby(specialty_col).agg({
                'is_upcoded_header': ['sum', 'count', 'mean']
            })
            specialty_upcode.columns = ['Upcoded', 'Total', 'Rate']
            specialty_upcode['Rate'] = (specialty_upcode['Rate'] * 100).round(2)
            specialty_upcode = specialty_upcode.sort_values('Rate', ascending=False)
            print(specialty_upcode.head(10))
        else:
            print("Provider taxonomy not available; skipping taxonomy-based analysis.")
            specialty_upcode = pd.DataFrame()

        # Provider volume analysis
        if {'provider_id', 'claim_id'}.issubset(self.df.columns):
            provider_volume = self.df.groupby('provider_id').agg({
                'claim_id': 'count',
                'is_upcoded_header': 'sum',
                'total_charges_header': 'sum' if 'total_charges_header' in self.df.columns else 'count'
            })
            provider_volume.columns = ['Claims', 'Upcoded', 'Total_Charges']
            provider_volume['Upcode_Rate'] = (provider_volume['Upcoded'] / provider_volume['Claims'] * 100).round(2)
            print(f"\n--- High-Volume Providers (Top 10) ---")
            print(provider_volume.nlargest(10, 'Claims'))
            print(f"\n--- High Upcoding Rate Providers (min 10 claims) ---")
            high_upcoders = provider_volume[provider_volume['Claims'] >= 10].nlargest(10, 'Upcode_Rate')
            print(high_upcoders)
        else:
            provider_volume = pd.DataFrame()

        return specialty_upcode, provider_volume

    # -----------------------
    # Service & clinical analysis
    # -----------------------
    def service_analysis(self):
        print("\n" + "="*80)
        print("6. SERVICE & CLINICAL ANALYSIS")
        print("="*80)

        if 'place_of_service' in self.df.columns:
            print("\n--- Place of Service Distribution ---")
            print(self.df['place_of_service'].value_counts())

        if 'claim_type' in self.df.columns:
            print("\n--- Claim Sub-Type Distribution ---")
            print(self.df['claim_type'].value_counts())

        if 'procedure_code' in self.df.columns:
            print("\n--- Top 10 Procedure Codes ---")
            print(self.df['procedure_code'].value_counts().head(10))

        # Length of stay analysis
        if 'length_of_stay' in self.df.columns:
            print("\n--- Length of Stay Statistics ---")
            print(self.df['length_of_stay'].describe())
            los_upcode = self.df.groupby('is_upcoded_header')['length_of_stay'].mean()
            print(f"\nAvg LOS (Upcoded): {los_upcode.get(True, 0):.2f}")
            print(f"Avg LOS (Clean):   {los_upcode.get(False, 0):.2f}")

        return self.df['place_of_service'].value_counts() if 'place_of_service' in self.df.columns else None

    # -----------------------
    # Diagnosis analysis
    # -----------------------
    def diagnosis_analysis(self):
        print("\n" + "="*80)
        print("7. DIAGNOSIS ANALYSIS")
        print("="*80)

        if 'primary_diagnosis' in self.df.columns:
            print("\n--- Top 15 Primary Diagnoses ---")
            print(self.df['primary_diagnosis'].value_counts().head(15))

        # Secondary diagnosis distribution across Diagnosis_Code_1..8
        if self.secondary_dx_cols:
            print("\n--- Top 15 Secondary Diagnoses (across Diagnosis_Code_1..8) ---")
            stacked = pd.Series(dtype=object)
            for c in self.secondary_dx_cols:
                stacked = pd.concat([stacked, self.df[c].dropna().astype(str)])
            print(stacked.value_counts().head(15))

            # Complexity: any secondary diagnosis present
            has_secondary = self.df[self.secondary_dx_cols].notna().any(axis=1).sum()
            total = len(self.df)
            print("\n--- Diagnosis Complexity ---")
            print(f"Rows with Secondary Diagnosis: {has_secondary:,} ({(has_secondary/total*100 if total else 0):.2f}%)")

            secondary_mask = self.df[self.secondary_dx_cols].notna().any(axis=1)
            no_secondary_mask = ~secondary_mask
            upcode_with_secondary = self.df.loc[secondary_mask, 'is_upcoded_header'].mean() * 100 if secondary_mask.any() else 0
            upcode_without_secondary = self.df.loc[no_secondary_mask, 'is_upcoded_header'].mean() * 100 if no_secondary_mask.any() else 0
            print(f"\nUpcode Rate (with secondary dx): {upcode_with_secondary:.2f}%")
            print(f"Upcode Rate (without secondary dx): {upcode_without_secondary:.2f}%")

        return self.df['primary_diagnosis'].value_counts() if 'primary_diagnosis' in self.df.columns else None

    # -----------------------
    # Temporal analysis
    # -----------------------
    def temporal_analysis(self):
        print("\n" + "="*80)
        print("8. TEMPORAL ANALYSIS")
        print("="*80)

        if 'service_date_header' in self.df.columns:
            print("\n--- Service Date Range ---")
            print(f"Earliest: {self.df['service_date_header'].min()}")
            print(f"Latest:   {self.df['service_date_header'].max()}")

        if 'year_month' in self.df.columns:
            monthly_claims = self.df.groupby('year_month').size()
            print(f"\n--- Monthly Claim Volume ---")
            print(monthly_claims)

            monthly_upcode = self.df.groupby('year_month')['is_upcoded_header'].mean() * 100
            print(f"\n--- Monthly Upcoding Rate ---")
            print(monthly_upcode)
            return monthly_claims
        return None

    # -----------------------
    # Correlation analysis
    # -----------------------
    def correlation_analysis(self):
        print("\n" + "="*80)
        print("9. CORRELATION ANALYSIS")
        print("="*80)

        # Select numerical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if 'is_upcoded_header' in self.df.columns and num_cols:
            upcode_corr = self.df[num_cols].corrwith(self.df['is_upcoded_header'].astype(int))
            upcode_corr = upcode_corr.sort_values(ascending=False)
            print("\n--- Correlation with Header-Level Upcoding ---")
            print(upcode_corr.head(25))
            return upcode_corr
        return None

    # -----------------------
    # Outlier detection
    # -----------------------
    def outlier_detection(self):
        print("\n" + "="*80)
        print("10. OUTLIER DETECTION")
        print("="*80)

        if 'total_charges_header' not in self.df.columns:
            print("Header charges not available; skipping outlier detection.")
            return pd.DataFrame()

        q1 = self.df['total_charges_header'].quantile(0.25)
        q3 = self.df['total_charges_header'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (self.df['total_charges_header'] < lower_bound) | (self.df['total_charges_header'] > upper_bound)
        charge_outliers = self.df[outlier_mask]
        non_outliers = self.df[~outlier_mask]

        print(f"\n--- Charge Outliers ---")
        print(f"Number of outliers: {len(charge_outliers):,}")
        print(f"Percentage: {(len(charge_outliers)/len(self.df)*100 if len(self.df) else 0):.2f}%")
        if 'is_upcoded_header' in self.df.columns and len(charge_outliers) > 0:
            print(f"Outlier upcoding rate: {charge_outliers['is_upcoded_header'].mean()*100:.2f}%")
            print(f"Non-outlier upcoding rate: {non_outliers['is_upcoded_header'].mean()*100:.2f}%")

        # LOS outliers
        if 'length_of_stay' in self.df.columns:
            los_threshold = self.df['length_of_stay'].quantile(0.95)
            los_outlier_mask = self.df['length_of_stay'] > los_threshold
            los_outliers = self.df[los_outlier_mask]
            print(f"\n--- Length of Stay Outliers (>95th percentile) ---")
            print(f"Number: {len(los_outliers):,}")
            if len(los_outliers) > 0 and 'is_upcoded_header' in self.df.columns:
                print(f"Avg LOS: {los_outliers['length_of_stay'].mean():.2f} days")
                print(f"Upcode rate: {los_outliers['is_upcoded_header'].mean()*100:.2f}%")
        return charge_outliers

    # -----------------------
    # Executive summary
    # -----------------------
    def generate_summary_report(self):
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)

        total_claims = self.df['claim_id'].nunique() if 'claim_id' in self.df.columns else np.nan
        total_lines = len(self.df)
        if 'claim_id' in self.df.columns and 'is_upcoded_header' in self.df.columns:
            upcoded_claims = self.df.groupby('claim_id')['is_upcoded_header'].first().sum()
            upcode_rate = (upcoded_claims / total_claims * 100) if total_claims else np.nan
        else:
            upcoded_claims, upcode_rate = np.nan, np.nan

        total_charges = self.df['total_charges_header'].sum() if 'total_charges_header' in self.df.columns else np.nan
        upcoded_charges = self.df.loc[self.df.get('is_upcoded_header', False) == True, 'total_charges_header'].sum() if 'total_charges_header' in self.df.columns else np.nan

        date_min = self.df['service_date_header'].min() if 'service_date_header' in self.df.columns else None
        date_max = self.df['service_date_header'].max() if 'service_date_header' in self.df.columns else None

        most_common_tax = self.df['provider_taxonomy'].mode()[0] if 'provider_taxonomy' in self.df.columns and not self.df['provider_taxonomy'].mode().empty else 'N/A'

        print(f"""
Dataset Overview:
- Total Claims: {total_claims:,} if not pd.isna(total_claims) else 'N/A'
- Total Line Items: {total_lines:,}
- Date Range: {date_min} to {date_max}
Upcoding Metrics:
- Upcoded Claims: {upcoded_claims:,} ({upcode_rate:.2f}% ) if not pd.isna(upcode_rate) else 'N/A'
- Financial Impact (charges): ${upcoded_charges:,.2f} ({(upcoded_charges/total_charges*100 if total_charges else 0):.2f}% of total header charges)
Key Findings:
- Unique Members: {self.df['member_id'].nunique():,} if 'member_id' in self.df.columns else 'N/A'
- Unique Providers: {self.df['provider_id'].nunique():,} if 'provider_id' in self.df.columns else 'N/A'
- Average Claim Charge (header): ${self.df['total_charges_header'].mean():,.2f} if 'total_charges_header' in self.df.columns else 'N/A'
- Most Common Provider Taxonomy: {most_common_tax}
""")

    # -----------------------
    # Visualizations
    # -----------------------
    def create_visualizations(self):
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        viz_dir = f"{self.output_dir}/visualizations"

        # 1. Upcoding Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Upcoding Analysis Overview', fontsize=16, fontweight='bold')

        # Header vs Line Upcoding
        if 'is_upcoded_header' in self.df.columns:
            upcoded = int(self.df['is_upcoded_header'].sum())
            clean = int((~self.df['is_upcoded_header']).sum())
        else:
            upcoded = clean = 0
        upcode_counts = pd.DataFrame({
            'Level': ['Header'],
            'Upcoded': [upcoded],
            'Clean': [clean]
        })
        upcode_counts.set_index('Level')[['Upcoded', 'Clean']].plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
        axes[0, 0].set_title('Upcoded vs Clean (by rows)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend(loc='best')
        axes[0, 0].tick_params(axis='x', rotation=0)

        # Upcoding Types Distribution
        if 'label_type' in self.df.columns:
            upcode_types = self.df['label_type'].value_counts().head(10)
            upcode_types.plot(kind='barh', ax=axes[0, 1], color='#3498db')
            axes[0, 1].set_title('Top 10 Label Types')
            axes[0, 1].set_xlabel('Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'Label_Type not available', ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')

        # Financial Impact (Header Charges split by label)
        if 'is_upcoded_header' in self.df.columns and 'total_charges_header' in self.df.columns:
            financial_data = pd.DataFrame({
                'Category': ['Upcoded', 'Clean'],
                'Total Charges': [
                    self.df.loc[self.df['is_upcoded_header'] == True, 'total_charges_header'].sum(),
                    self.df.loc[self.df['is_upcoded_header'] == False, 'total_charges_header'].sum(),
                ]
            })
            axes[1, 0].pie(financial_data['Total Charges'], labels=financial_data['Category'],
                           autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], startangle=90)
            axes[1, 0].set_title('Financial Impact Distribution (Header Charges)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for pie chart', ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')

        # Upcoding Rate by Month
        if 'year_month' in self.df.columns and 'is_upcoded_header' in self.df.columns:
            monthly_upcode = self.df.groupby('year_month')['is_upcoded_header'].mean() * 100
            monthly_upcode.plot(kind='line', ax=axes[1, 1], marker='o', color='#e74c3c', linewidth=2)
            axes[1, 1].set_title('Monthly Upcoding Rate Trend')
            axes[1, 1].set_ylabel('Upcoding Rate (%)')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Temporal data not available', ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/01_upcoding_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 01_upcoding_overview.png")

        # 2. Provider Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Provider Analysis', fontsize=16, fontweight='bold')

        # Top Taxonomies by Volume
        if 'provider_taxonomy' in self.df.columns:
            top_tax = self.df['provider_taxonomy'].value_counts().head(15)
            top_tax.plot(kind='barh', ax=axes[0, 0], color='#9b59b6')
            axes[0, 0].set_title('Top 15 Provider Taxonomies by Volume')
            axes[0, 0].set_xlabel('Number of Rows')
        else:
            axes[0, 0].text(0.5, 0.5, 'Provider taxonomy not available', ha='center', va='center', fontsize=12)
            axes[0, 0].axis('off')

        # Upcoding Rate by Taxonomy
        if {'provider_taxonomy', 'is_upcoded_header'}.issubset(self.df.columns):
            specialty_upcode = self.df.groupby('provider_taxonomy').agg({'is_upcoded_header': ['sum', 'count', 'mean']})
            specialty_upcode.columns = ['Upcoded', 'Total', 'Rate']
            specialty_upcode = specialty_upcode[specialty_upcode['Total'] >= 100]
            specialty_upcode['Rate'] = specialty_upcode['Rate'] * 100
            specialty_upcode.nlargest(15, 'Rate')['Rate'].plot(kind='barh', ax=axes[0, 1], color='#e67e22')
            axes[0, 1].set_title('Top 15 Taxonomies by Upcoding Rate (min 100 rows)')
            axes[0, 1].set_xlabel('Upcoding Rate (%)')
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')

        # Provider Volume Distribution
        if 'provider_id' in self.df.columns:
            provider_volume = self.df.groupby('provider_id').size()
            axes[1, 0].hist(provider_volume, bins=50, color='#1abc9c', edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Provider Volume Distribution (rows per provider)')
            axes[1, 0].set_xlabel('Rows per Provider')
            axes[1, 0].set_ylabel('Number of Providers')
            axes[1, 0].set_yscale('log')
        else:
            axes[1, 0].text(0.5, 0.5, 'Provider_ID not available', ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')

        # Place of Service Distribution
        if 'place_of_service' in self.df.columns:
            pos_dist = self.df['place_of_service'].value_counts().head(10)
            pos_dist.plot(kind='bar', ax=axes[1, 1], color='#34495e')
            axes[1, 1].set_title('Top 10 Places of Service')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'Place_Of_Service_Indicator not available', ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/02_provider_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 02_provider_analysis.png")

        # 3. Financial Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial Analysis', fontsize=16, fontweight='bold')

        # Charge Distribution (header)
        if 'total_charges_header' in self.df.columns:
            charges = self.df['total_charges_header'].dropna()
            axes[0, 0].hist(charges[charges < charges.quantile(0.95)], bins=50,
                            color='#16a085', edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Header Charges Distribution (up to 95th percentile)')
            axes[0, 0].set_xlabel('Total Charges ($)')
            axes[0, 0].set_ylabel('Frequency')

            # Box plot: Charges by Upcoding Status
            if 'is_upcoded_header' in self.df.columns:
                upcode_charges_data = [
                    self.df.loc[self.df['is_upcoded_header'] == False, 'total_charges_header'].dropna(),
                    self.df.loc[self.df['is_upcoded_header'] == True, 'total_charges_header'].dropna()
                ]
                bp = axes[0, 1].boxplot(upcode_charges_data, labels=['Clean', 'Upcoded'], patch_artist=True)
                for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
                    patch.set_facecolor(color)
                axes[0, 1].set_title('Header Charges by Upcoding Status')
                axes[0, 1].set_ylabel('Total Charges ($)')
                axes[0, 1].set_yscale('log')
        else:
            axes[0, 0].text(0.5, 0.5, 'Header charges not available', ha='center', va='center', fontsize=12)
            axes[0, 0].axis('off')

        # Payment vs Charges (header)
        if {'total_charges_header', 'expected_payment_header'}.issubset(self.df.columns):
            sample_df = self.df.sample(min(10000, len(self.df))) if len(self.df) > 0 else self.df
            axes[1, 0].scatter(sample_df['total_charges_header'], sample_df['expected_payment_header'],
                               alpha=0.3, s=10, color='#8e44ad')
            axes[1, 0].set_title('Expected Payment vs Total Charges (header, sample)')
            axes[1, 0].set_xlabel('Total Charges ($)')
            axes[1, 0].set_ylabel('Expected Payment ($)')
            if len(sample_df) > 0:
                max_val = sample_df['total_charges_header'].max()
                axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 line')
                axes[1, 0].legend()

        # Charge-to-Payment Ratio (header)
        if 'charge_to_payment_ratio' in self.df.columns:
            ratio = self.df['charge_to_payment_ratio'].dropna()
            ratio_filtered = ratio[(ratio > 0) & (ratio < ratio.quantile(0.95))]
            axes[1, 1].hist(ratio_filtered, bins=50, color='#c0392b', edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Charge-to-Payment Ratio Distribution (Header)')
            axes[1, 1].set_xlabel('Ratio')
            axes[1, 1].set_ylabel('Frequency')
        else:
            axes[1, 1].text(0.5, 0.5, 'Ratio not computed', ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/03_financial_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 03_financial_analysis.png")

        # 4. Clinical & Procedure Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clinical & Procedure Analysis', fontsize=16, fontweight='bold')

        # Top Primary Diagnoses
        if 'primary_diagnosis' in self.df.columns:
            top_dx = self.df['primary_diagnosis'].value_counts().head(15)
            top_dx.plot(kind='barh', ax=axes[0, 0], color='#27ae60')
            axes[0, 0].set_title('Top 15 Primary Diagnoses')
            axes[0, 0].set_xlabel('Count')
        else:
            axes[0, 0].text(0.5, 0.5, 'Primary_Diagnosis_Code not available', ha='center', va='center', fontsize=12)
            axes[0, 0].axis('off')

        # Top Procedure Codes
        if 'procedure_code' in self.df.columns:
            top_proc = self.df['procedure_code'].value_counts().head(15)
            top_proc.plot(kind='barh', ax=axes[0, 1], color='#2980b9')
            axes[0, 1].set_title('Top 15 Procedure Codes')
            axes[0, 1].set_xlabel('Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'Procedure_Code_ID not available', ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')

        # Length of Stay Distribution
        if 'length_of_stay' in self.df.columns:
            los = self.df['length_of_stay'].dropna()
            if len(los) > 0:
                los_filtered = los[los < los.quantile(0.95)]
                axes[1, 0].hist(los_filtered, bins=30, color='#f39c12', edgecolor='black', alpha=0.7)
                axes[1, 0].set_title('Length of Stay Distribution (up to 95th percentile)')
                axes[1, 0].set_xlabel('Days')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'No LOS data', ha='center', va='center', fontsize=12)
                axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Length of Stay data not available', ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')

        # Claim Sub-Type Distribution
        if 'claim_type' in self.df.columns:
            claim_types = self.df['claim_type'].value_counts()
            axes[1, 1].pie(claim_types.values, labels=claim_types.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Claim Sub-Type Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'Claim_Sub_Type not available', ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/04_clinical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: 04_clinical_analysis.png")

        # 5. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_cols = [
            'total_charges_header', 'expected_payment_header',
            'total_charges_line', 'units', 'unit_cost', 'length_of_stay', 'allowed_amount_line'
        ]
        corr_cols = [c for c in corr_cols if c in self.df.columns]
        if len(corr_cols) > 1:
            corr_matrix = self.df[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/05_correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: 05_correlation_heatmap.png")
        else:
            plt.close()
            print("Skipped correlation heatmap (insufficient numeric columns)")

        print("\n✓ All visualizations generated successfully!")

    # -----------------------
    # CSV Exports
    # -----------------------
    def export_summary_data(self):
        print("\n" + "="*80)
        print("EXPORTING SUMMARY DATA TO CSV")
        print("="*80)
        csv_dir = f"{self.output_dir}/csv_outputs"

        # 1. Overall Summary Statistics
        summary_metrics = {
            'Total Claims': self.df['claim_id'].nunique() if 'claim_id' in self.df.columns else np.nan,
            'Total Line Items': len(self.df),
            'Unique Members': self.df['member_id'].nunique() if 'member_id' in self.df.columns else np.nan,
            'Unique Providers': self.df['provider_id'].nunique() if 'provider_id' in self.df.columns else np.nan,
            'Upcoded Claims (Header)': self.df.groupby('claim_id')['is_upcoded_header'].first().sum() if {'claim_id','is_upcoded_header'}.issubset(self.df.columns) else np.nan,
            'Upcoding Rate (Header) %': self.df.groupby('claim_id')['is_upcoded_header'].first().mean()*100 if {'claim_id','is_upcoded_header'}.issubset(self.df.columns) else np.nan,
            'Upcoded Rows': self.df['is_upcoded_header'].sum() if 'is_upcoded_header' in self.df.columns else np.nan,
            'Upcoding Rate (Row) %': self.df['is_upcoded_header'].mean()*100 if 'is_upcoded_header' in self.df.columns else np.nan,
            'Total Header Charges': self.df['total_charges_header'].sum() if 'total_charges_header' in self.df.columns else np.nan,
            'Upcoded Header Charges': self.df.loc[self.df.get('is_upcoded_header', False) == True, 'total_charges_header'].sum() if 'total_charges_header' in self.df.columns else np.nan,
            'Average Header Charge': self.df['total_charges_header'].mean() if 'total_charges_header' in self.df.columns else np.nan,
            'Median Header Charge': self.df['total_charges_header'].median() if 'total_charges_header' in self.df.columns else np.nan,
            'Average Expected Payment (header)': self.df['expected_payment_header'].mean() if 'expected_payment_header' in self.df.columns else np.nan,
        }
        summary_stats = pd.DataFrame({
            'Metric': list(summary_metrics.keys()),
            'Value': list(summary_metrics.values())
        })
        summary_stats.to_csv(f"{csv_dir}/01_overall_summary.csv", index=False)
        print(f"✓ Saved: 01_overall_summary.csv")

        # 2. Upcoding Type Analysis (Label_Type)
        if 'label_type' in self.df.columns:
            upcode_type_summary = self.df['label_type'].value_counts()
            upcode_type_summary.to_csv(f"{csv_dir}/02_upcoding_types.csv", header=['Count'])
            print(f"✓ Saved: 02_upcoding_types.csv")

        # 3. Procedure Codes
        if 'procedure_code' in self.df.columns:
            proc_summary = self.df['procedure_code'].value_counts().head(50)
            proc_summary.to_csv(f"{csv_dir}/03_procedure_codes.csv", header=['Count'])
            print(f"✓ Saved: 03_procedure_codes.csv")

        # 4. Provider Taxonomy Analysis
        if {'provider_taxonomy', 'is_upcoded_header', 'claim_id'}.issubset(self.df.columns):
            specialty_analysis = self.df.groupby('provider_taxonomy').agg({
                'claim_id': 'count',
                'is_upcoded_header': ['sum', 'mean'],
                'total_charges_header': ['sum', 'mean'] if 'total_charges_header' in self.df.columns else 'count'
            }).round(2)
            # rename columns
            if isinstance(specialty_analysis.columns, pd.MultiIndex):
                specialty_analysis.columns = ['Total_Rows', 'Upcoded_Count', 'Upcode_Rate', 'Total_Charges', 'Avg_Charge'][:len(specialty_analysis.columns)]
            if 'Upcode_Rate' in specialty_analysis.columns:
                specialty_analysis['Upcode_Rate'] = (specialty_analysis['Upcode_Rate'] * 100).round(2)
            specialty_analysis = specialty_analysis.sort_values('Total_Rows', ascending=False)
            specialty_analysis.to_csv(f"{csv_dir}/04_taxonomy_analysis.csv")
            print(f"✓ Saved: 04_taxonomy_analysis.csv")

        # 5. Provider-Level Analysis (Top 1000)
        prov_cols = ['provider_id', 'claim_id', 'is_upcoded_header']
        if all(c in self.df.columns for c in prov_cols):
            agg_dict = {
                'claim_id': 'count',
                'is_upcoded_header': ['sum', 'mean']
            }
            if 'total_charges_header' in self.df.columns:
                agg_dict['total_charges_header'] = ['sum', 'mean']
            if 'provider_taxonomy' in self.df.columns:
                agg_dict['provider_taxonomy'] = 'first'
            if 'provider_npi' in self.df.columns:
                agg_dict['provider_npi'] = 'first'
            provider_analysis = self.df.groupby('provider_id').agg(agg_dict).round(2)
            if isinstance(provider_analysis.columns, pd.MultiIndex):
                new_cols = []
                for a, b in provider_analysis.columns:
                    if b == '':
                        new_cols.append(a)
                    else:
                        new_cols.append({'sum': 'Sum', 'mean': 'Avg', 'count': 'Count'}.get(b, b).join([' ']) if False else f"{a if a not in ['is_upcoded_header'] else 'Upcoded'}_{b.capitalize() if b else ''}".strip('_'))
                provider_analysis.columns = new_cols
            # friendly column names
            provider_analysis.rename(columns={
                'claim_id_Count': 'Total_Rows',
                'is_upcoded_header_Sum': 'Upcoded_Count',
                'is_upcoded_header_Mean': 'Upcode_Rate',
                'total_charges_header_Sum': 'Total_Charges',
                'total_charges_header_Mean': 'Avg_Charge',
            }, inplace=True)
            if 'Upcode_Rate' in provider_analysis.columns:
                provider_analysis['Upcode_Rate'] = (provider_analysis['Upcode_Rate'] * 100).round(2)
            provider_analysis = provider_analysis.sort_values('Total_Rows', ascending=False).head(1000)
            provider_analysis.to_csv(f"{csv_dir}/05_top_providers.csv")
            print(f"✓ Saved: 05_top_providers.csv")

        # 6. Diagnosis Codes
        if 'primary_diagnosis' in self.df.columns or self.secondary_dx_cols:
            primary = self.df['primary_diagnosis'].value_counts().head(100) if 'primary_diagnosis' in self.df.columns else pd.Series(dtype=int)
            secondary_counts = pd.Series(dtype=int)
            for c in self.secondary_dx_cols:
                secondary_counts = secondary_counts.add(self.df[c].value_counts(), fill_value=0)
            secondary_counts = secondary_counts.sort_values(ascending=False).head(100)
            dx_df = pd.DataFrame({'Primary_Diagnosis': primary, 'Secondary_Diagnosis': secondary_counts}).fillna(0).astype(int)
            dx_df.to_csv(f"{csv_dir}/06_diagnosis_codes.csv")
            print(f"✓ Saved: 06_diagnosis_codes.csv")

        # 7. Place of Service
        if 'place_of_service' in self.df.columns:
            pos_analysis = self.df.groupby('place_of_service').agg({
                'claim_id': 'count' if 'claim_id' in self.df.columns else 'size',
                'is_upcoded_header': ['sum', 'mean'] if 'is_upcoded_header' in self.df.columns else 'size',
                'total_charges_header': 'mean' if 'total_charges_header' in self.df.columns else 'size'
            }).round(2)
            if isinstance(pos_analysis.columns, pd.MultiIndex):
                pos_analysis.columns = ['Total_Rows', 'Upcoded_Count', 'Upcode_Rate', 'Avg_Charge'][:len(pos_analysis.columns)]
            if 'Upcode_Rate' in pos_analysis.columns:
                pos_analysis['Upcode_Rate'] = (pos_analysis['Upcode_Rate'] * 100).round(2)
            pos_analysis = pos_analysis.sort_values('Total_Rows', ascending=False)
            pos_analysis.to_csv(f"{csv_dir}/07_place_of_service.csv")
            print(f"✓ Saved: 07_place_of_service.csv")

        # 8. Monthly Trends
        if 'year_month' in self.df.columns:
            monthly_trends = self.df.groupby('year_month').agg({
                'claim_id': 'count' if 'claim_id' in self.df.columns else 'size',
                'is_upcoded_header': ['sum', 'mean'] if 'is_upcoded_header' in self.df.columns else 'size',
                'total_charges_header': ['sum', 'mean'] if 'total_charges_header' in self.df.columns else 'size'
            }).round(2)
            if isinstance(monthly_trends.columns, pd.MultiIndex):
                monthly_trends.columns = ['Total_Rows', 'Upcoded_Count', 'Upcode_Rate', 'Total_Charges', 'Avg_Charge'][:len(monthly_trends.columns)]
            if 'Upcode_Rate' in monthly_trends.columns:
                monthly_trends['Upcode_Rate'] = (monthly_trends['Upcode_Rate'] * 100).round(2)
            monthly_trends.to_csv(f"{csv_dir}/08_monthly_trends.csv")
            print(f"✓ Saved: 08_monthly_trends.csv")

        # 9. Financial Metrics Summary
        financial_vals = {}
        if 'total_charges_header' in self.df.columns:
            financial_vals.update({
                'Total Header Charges': self.df['total_charges_header'].sum(),
                'Mean Header Charges': self.df['total_charges_header'].mean(),
                'Median Header Charges': self.df['total_charges_header'].median(),
                'Std Header Charges': self.df['total_charges_header'].std(),
            })
        if 'expected_payment_header' in self.df.columns:
            financial_vals.update({
                'Total Expected Payment (header)': self.df['expected_payment_header'].sum(),
                'Mean Expected Payment (header)': self.df['expected_payment_header'].mean(),
            })
        if 'allowed_amount_line' in self.df.columns:
            financial_vals.update({
                'Total Allowed Amount (line)': self.df['allowed_amount_line'].sum(),
                'Mean Allowed Amount (line)': self.df['allowed_amount_line'].mean(),
            })
        if financial_vals:
            financial_summary = pd.DataFrame({'Metric': list(financial_vals.keys()), 'Value': list(financial_vals.values())}).round(2)
            financial_summary.to_csv(f"{csv_dir}/09_financial_summary.csv", index=False)
            print(f"✓ Saved: 09_financial_summary.csv")

        # 10. High-Risk Rows (High charges + Upcoded)
        if {'is_upcoded_header', 'total_charges_header'}.issubset(self.df.columns):
            high_risk = self.df[
                (self.df['is_upcoded_header'] == True) &
                (self.df['total_charges_header'] > self.df['total_charges_header'].quantile(0.90))
            ][[
                c for c in [
                    'claim_id', 'member_id', 'provider_id', 'provider_taxonomy',
                    'total_charges_header', 'expected_payment_header', 'label_type',
                    'primary_diagnosis', 'procedure_code'
                ] if c in self.df.columns
            ]].drop_duplicates(subset=['claim_id'] if 'claim_id' in self.df.columns else None).head(1000)
            high_risk.to_csv(f"{csv_dir}/10_high_risk_claims.csv", index=False)
            print(f"✓ Saved: 10_high_risk_claims.csv")

        # 11. Correlation Matrix
        corr_cols = [
            'total_charges_header', 'expected_payment_header',
            'total_charges_line', 'units', 'length_of_stay', 'allowed_amount_line', 'unit_cost'
        ]
        corr_cols = [c for c in corr_cols if c in self.df.columns]
        if len(corr_cols) > 1:
            corr_matrix = self.df[corr_cols].corr().round(3)
            corr_matrix.to_csv(f"{csv_dir}/11_correlation_matrix.csv")
            print(f"✓ Saved: 11_correlation_matrix.csv")

        print("\n✓ All CSV outputs exported successfully!")
        print(f"\nOutput location: {os.path.abspath(self.output_dir)}")

    # -----------------------
    # Pipeline
    # -----------------------
    def run_full_analysis(self):
        print("\n" + "="*80)
        print("HEALTHCARE CLAIMS DESCRIPTIVE ANALYSIS")
        print("Focus: Upcoding Detection (Label)")
        print("="*80)

        self.basic_statistics()
        self.claims_overview()
        self.financial_analysis()
        self.upcoding_analysis()
        self.provider_analysis()
        self.service_analysis()
        self.diagnosis_analysis()
        self.temporal_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        self.generate_summary_report()

        # Generate visualizations and exports
        self.create_visualizations()
        self.export_summary_data()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {os.path.abspath(self.output_dir)}")
        print(f" - Visualizations: {os.path.abspath(self.output_dir)}/visualizations/")
        print(f" - CSV Outputs: {os.path.abspath(self.output_dir)}/csv_outputs/")

    def close(self):
        """Close database connection"""
        self.conn.close()


# Usage Example
if __name__ == "__main__":
    # Initialize analyzer with output directory
    analyzer = HealthcareClaimsAnalyzer(
        db_path='claims_database.db',
        output_dir='claims_analysis_results'
    )

    # Load data from the specified table
    analyzer.load_data(table_name='ClaimsData')

    # Run full analysis with visualizations and CSV exports
    analyzer.run_full_analysis()

    # Close connection
    analyzer.close()

    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("""
    To use this script:
      1) Ensure the SQLite DB file 'claims_database.db' exists with table 'ClaimsData'.
      2) Confirm your upcoding indicator is in field 'Label'. Common truthy values
         (1, yes, true, y, upcoded) are treated as upcoded; others as clean.
      3) Run:  python 01_Claims_Descritiptive_Analysis_SQLite.py

    Outputs are generated in 'claims_analysis_results/' directory:
      VISUALIZATIONS (PNG files):
        - 01_upcoding_overview.png
        - 02_provider_analysis.png
        - 03_financial_analysis.png
        - 04_clinical_analysis.png
        - 05_correlation_heatmap.png (if enough numeric columns)

      CSV OUTPUTS:
        - 01_overall_summary.csv
        - 02_upcoding_types.csv (if Label_Type available)
        - 03_procedure_codes.csv (if Procedure_Code_ID available)
        - 04_taxonomy_analysis.csv
        - 05_top_providers.csv
        - 06_diagnosis_codes.csv
        - 07_place_of_service.csv
        - 08_monthly_trends.csv
        - 09_financial_summary.csv
        - 10_high_risk_claims.csv
        - 11_correlation_matrix.csv
    """)
