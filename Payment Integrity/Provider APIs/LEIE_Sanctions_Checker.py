"""
LEIE Exclusions & Sanctions Identification System
Checks providers in claims database against HHS-OIG LEIE database
"""

import sqlite3
import requests
import pandas as pd
import zipfile
import io
import os
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LEIEDownloader:
    """Download and parse LEIE database from HHS-OIG"""
    
    LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
    
    def __init__(self, cache_dir: str = "./leie_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_leie(self) -> pd.DataFrame:
        """Download latest LEIE database"""
        logger.info("Downloading LEIE database from HHS-OIG...")
        
        try:
            response = requests.get(self.LEIE_URL, timeout=60)
            response.raise_for_status()
            
            # Save to cache
            cache_file = os.path.join(
                self.cache_dir, 
                f"leie_{datetime.now().strftime('%Y%m%d')}.csv"
            )
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Downloaded {len(df)} LEIE records")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading LEIE: {e}")
            # Try to load from cache
            return self._load_from_cache()
    
    def _load_from_cache(self) -> pd.DataFrame:
        """Load most recent cached LEIE file"""
        cache_files = sorted([
            f for f in os.listdir(self.cache_dir) 
            if f.startswith('leie_') and f.endswith('.csv')
        ], reverse=True)
        
        if cache_files:
            cache_file = os.path.join(self.cache_dir, cache_files[0])
            logger.info(f"Loading cached LEIE from {cache_file}")
            return pd.read_csv(cache_file)
        else:
            raise FileNotFoundError("No cached LEIE data available")


class LEIEMatcher:
    """Match providers against LEIE database"""
    
    def __init__(self, leie_df: pd.DataFrame):
        self.leie_df = leie_df
        self._prepare_leie_data()
        
    def _prepare_leie_data(self):
        """Prepare LEIE data for matching"""
        # Standardize column names (LEIE uses different column names)
        # Common columns: LASTNAME, FIRSTNAME, MIDNAME, BUSNAME, NPI, UPIN, DOB, ADDRESS, CITY, STATE, ZIP, EXCLTYPE, EXCLDATE, REINDATE, WAIVERDATE, WAIVERSTATE
        
        # Create NPI lookup
        if 'NPI' in self.leie_df.columns:
            self.npi_exclusions = set(
                self.leie_df[self.leie_df['NPI'].notna()]['NPI'].astype(str).str.strip()
            )
            logger.info(f"Indexed {len(self.npi_exclusions)} NPI exclusions")
        else:
            self.npi_exclusions = set()
            
        # Create name lookup for providers without NPI
        self.name_exclusions = {}
        if 'LASTNAME' in self.leie_df.columns and 'FIRSTNAME' in self.leie_df.columns:
            for _, row in self.leie_df.iterrows():
                if pd.notna(row.get('LASTNAME')):
                    key = self._create_name_key(
                        row.get('LASTNAME', ''),
                        row.get('FIRSTNAME', ''),
                        row.get('MIDNAME', '')
                    )
                    self.name_exclusions[key] = row.to_dict()
    
    def _create_name_key(self, last: str, first: str, middle: str = '') -> str:
        """Create standardized name key"""
        parts = [str(last).upper().strip(), str(first).upper().strip()]
        if middle and str(middle).strip():
            parts.append(str(middle).upper().strip()[0])  # First initial only
        return '|'.join(parts)
    
    def check_npi(self, npi: str) -> Tuple[bool, Dict]:
        """Check if NPI is excluded"""
        if not npi or pd.isna(npi):
            return False, {}
            
        npi_clean = str(npi).strip()
        if npi_clean in self.npi_exclusions:
            # Get full exclusion details
            match = self.leie_df[self.leie_df['NPI'] == npi_clean].iloc[0]
            return True, {
                'match_type': 'NPI',
                'npi': npi_clean,
                'name': f"{match.get('FIRSTNAME', '')} {match.get('LASTNAME', '')}",
                'exclusion_type': match.get('EXCLTYPE', ''),
                'exclusion_date': match.get('EXCLDATE', ''),
                'reinstatement_date': match.get('REINDATE', ''),
                'waiver_state': match.get('WAIVERSTATE', '')
            }
        return False, {}
    
    def check_provider_name(self, provider_name: str) -> Tuple[bool, Dict]:
        """Check provider by name (fallback when no NPI)"""
        if not provider_name or pd.isna(provider_name):
            return False, {}
        
        # Try to parse name
        name_parts = str(provider_name).upper().strip().split(',')
        if len(name_parts) >= 2:
            last = name_parts[0].strip()
            first_parts = name_parts[1].strip().split()
            first = first_parts[0] if first_parts else ''
            middle = first_parts[1][0] if len(first_parts) > 1 else ''
            
            key = self._create_name_key(last, first, middle)
            if key in self.name_exclusions:
                match = self.name_exclusions[key]
                return True, {
                    'match_type': 'NAME',
                    'name': provider_name,
                    'exclusion_type': match.get('EXCLTYPE', ''),
                    'exclusion_date': match.get('EXCLDATE', ''),
                    'reinstatement_date': match.get('REINDATE', ''),
                    'waiver_state': match.get('WAIVERSTATE', '')
                }
        
        return False, {}


class ClaimsAnalyzer:
    """Analyze claims database for LEIE exclusions"""
    
    def __init__(self, db_path: str, matcher: LEIEMatcher):
        self.db_path = db_path
        self.matcher = matcher
        
    def get_unique_providers(self) -> pd.DataFrame:
        """Extract unique providers from claims database"""
        logger.info("Extracting unique providers from claims database...")
        
        query = """
        SELECT DISTINCT 
            Provider_NPI,
            Service_Provider_Name,
            Service_Provider_City,
            Service_Provider_State,
            Service_Provider_Zip,
            Provider_ID,
            Group_Name,
            Group_ID
        FROM ClaimsData
        WHERE Provider_NPI IS NOT NULL OR Service_Provider_Name IS NOT NULL
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        logger.info(f"Found {len(df)} unique providers")
        return df
    
    def scan_for_exclusions(self) -> pd.DataFrame:
        """Scan all providers for exclusions"""
        providers = self.get_unique_providers()
        results = []
        
        logger.info("Scanning providers against LEIE database...")
        for idx, provider in providers.iterrows():
            # Check by NPI first
            is_excluded, details = self.matcher.check_npi(provider['Provider_NPI'])
            
            # If not found by NPI, try name
            if not is_excluded and provider['Service_Provider_Name']:
                is_excluded, details = self.matcher.check_provider_name(
                    provider['Service_Provider_Name']
                )
            
            if is_excluded:
                result = {
                    'Provider_NPI': provider['Provider_NPI'],
                    'Provider_Name': provider['Service_Provider_Name'],
                    'Provider_ID': provider['Provider_ID'],
                    'City': provider['Service_Provider_City'],
                    'State': provider['Service_Provider_State'],
                    'Match_Type': details['match_type'],
                    'Exclusion_Type': details['exclusion_type'],
                    'Exclusion_Date': details['exclusion_date'],
                    'Reinstatement_Date': details.get('reinstatement_date', ''),
                    'Waiver_State': details.get('waiver_state', ''),
                    'Status': self._determine_status(details)
                }
                results.append(result)
                
                if len(results) % 10 == 0:
                    logger.info(f"Found {len(results)} exclusions so far...")
        
        logger.info(f"Scan complete. Found {len(results)} excluded providers")
        return pd.DataFrame(results)
    
    def _determine_status(self, details: Dict) -> str:
        """Determine if exclusion is active"""
        reinstate_date = details.get('reinstatement_date', '')
        if reinstate_date and reinstate_date.strip():
            try:
                reinstate = datetime.strptime(reinstate_date, '%Y%m%d')
                if reinstate <= datetime.now():
                    return 'REINSTATED'
            except:
                pass
        return 'ACTIVE_EXCLUSION'
    
    def flag_affected_claims(self, exclusions_df: pd.DataFrame) -> pd.DataFrame:
        """Flag claims associated with excluded providers"""
        if exclusions_df.empty:
            logger.info("No exclusions found, no claims to flag")
            return pd.DataFrame()
        
        excluded_npis = exclusions_df['Provider_NPI'].dropna().tolist()
        excluded_ids = exclusions_df['Provider_ID'].dropna().tolist()
        
        logger.info("Flagging affected claims...")
        
        placeholders_npi = ','.join(['?' for _ in excluded_npis])
        placeholders_id = ','.join(['?' for _ in excluded_ids])
        
        query = f"""
        SELECT 
            Claim_ID,
            Line_Sequence_No,
            Provider_NPI,
            Provider_ID,
            Service_Provider_Name,
            Claim_Service_Start_Date,
            Claim_Service_End_Date,
            Line_Paid_Amount,
            Claim_Total_Payable,
            Member_ID
        FROM ClaimsData
        WHERE Provider_NPI IN ({placeholders_npi})
           OR Provider_ID IN ({placeholders_id})
        ORDER BY Claim_Service_Start_Date DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=excluded_npis + excluded_ids)
        
        logger.info(f"Found {len(df)} claims affected by exclusions")
        return df


class ReportGenerator:
    """Generate exclusion reports"""
    
    @staticmethod
    def generate_summary_report(exclusions_df: pd.DataFrame, 
                                claims_df: pd.DataFrame) -> str:
        """Generate text summary report"""
        report = []
        report.append("="*70)
        report.append("LEIE EXCLUSIONS & SANCTIONS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*70)
        report.append("")
        
        # Exclusions summary
        report.append(f"TOTAL EXCLUDED PROVIDERS FOUND: {len(exclusions_df)}")
        
        if not exclusions_df.empty:
            active = len(exclusions_df[exclusions_df['Status'] == 'ACTIVE_EXCLUSION'])
            reinstated = len(exclusions_df[exclusions_df['Status'] == 'REINSTATED'])
            report.append(f"  - Active Exclusions: {active}")
            report.append(f"  - Reinstated: {reinstated}")
            report.append("")
            
            # By exclusion type
            report.append("EXCLUSIONS BY TYPE:")
            type_counts = exclusions_df['Exclusion_Type'].value_counts()
            for exc_type, count in type_counts.items():
                report.append(f"  - {exc_type}: {count}")
            report.append("")
        
        # Claims impact
        if not claims_df.empty:
            report.append(f"AFFECTED CLAIMS: {len(claims_df)}")
            total_paid = claims_df['Line_Paid_Amount'].sum()
            report.append(f"TOTAL AMOUNT PAID: ${total_paid:,.2f}")
            report.append("")
            
            # Claims by provider
            report.append("TOP 10 PROVIDERS BY CLAIM COUNT:")
            top_providers = claims_df.groupby('Service_Provider_Name').size().sort_values(ascending=False).head(10)
            for provider, count in top_providers.items():
                report.append(f"  - {provider}: {count} claims")
        
        report.append("="*70)
        return "\n".join(report)
    
    @staticmethod
    def save_reports(exclusions_df: pd.DataFrame, 
                     claims_df: pd.DataFrame,
                     output_dir: str = "./leie_reports"):
        """Save detailed reports to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save exclusions
        if not exclusions_df.empty:
            exc_file = os.path.join(output_dir, f"exclusions_{timestamp}.csv")
            exclusions_df.to_csv(exc_file, index=False)
            logger.info(f"Saved exclusions report: {exc_file}")
        
        # Save affected claims
        if not claims_df.empty:
            claims_file = os.path.join(output_dir, f"affected_claims_{timestamp}.csv")
            claims_df.to_csv(claims_file, index=False)
            logger.info(f"Saved affected claims report: {claims_file}")
        
        # Save summary
        summary = ReportGenerator.generate_summary_report(exclusions_df, claims_df)
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        logger.info(f"Saved summary report: {summary_file}")
        
        return summary


def main():
    """Main execution function"""
    # Configuration
    DB_PATH = "claims_database.db"
    CACHE_DIR = "./leie_cache"
    OUTPUT_DIR = "./leie_reports"
    
    try:
        # Step 1: Download LEIE database
        downloader = LEIEDownloader(cache_dir=CACHE_DIR)
        leie_df = downloader.download_leie()
        
        # Step 2: Initialize matcher
        matcher = LEIEMatcher(leie_df)
        
        # Step 3: Analyze claims database
        analyzer = ClaimsAnalyzer(DB_PATH, matcher)
        exclusions_df = analyzer.scan_for_exclusions()
        
        # Step 4: Flag affected claims
        affected_claims_df = analyzer.flag_affected_claims(exclusions_df)
        
        # Step 5: Generate reports
        summary = ReportGenerator.save_reports(
            exclusions_df, 
            affected_claims_df,
            output_dir=OUTPUT_DIR
        )
        
        # Print summary to console
        print("\n" + summary)
        
        # Return results for programmatic use
        return {
            'exclusions': exclusions_df,
            'affected_claims': affected_claims_df,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    results = main()