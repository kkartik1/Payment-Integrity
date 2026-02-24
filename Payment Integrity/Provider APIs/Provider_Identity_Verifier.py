"""
Provider Identity & Specialty Verification System
Uses NPPES NPI Registry API to validate provider information from claims data
"""

import sqlite3
import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time


@dataclass
class ProviderIdentity:
    """Data class to store provider identity information"""
    npi: str
    provider_type: str  # "Individual" or "Organization"
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    taxonomy_code: str = None
    taxonomy_description: str = None
    primary_specialty: str = None
    address: str = None
    city: str = None
    state: str = None
    zip_code: str = None
    phone: str = None
    enumeration_date: str = None
    last_updated: str = None
    status: str = None
    identity_established: bool = False
    verification_reasons: List[str] = None
    
    def __post_init__(self):
        if self.verification_reasons is None:
            self.verification_reasons = []


class NPPESAPIClient:
    """Client for NPPES NPI Registry API"""
    
    BASE_URL = "https://npiregistry.cms.hhs.gov/api/"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize NPPES API client
        
        Args:
            rate_limit_delay: Delay between API calls in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def lookup_npi(self, npi: str, version: str = "2.1") -> Optional[Dict]:
        """
        Look up provider by NPI number
        
        Args:
            npi: National Provider Identifier
            version: API version (default 2.1)
            
        Returns:
            Dictionary with provider information or None if not found
        """
        self._rate_limit()
        
        params = {
            "number": npi,
            "version": version
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result_count", 0) > 0:
                return data["results"][0]
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching NPI {npi}: {e}")
            return None
    
    def lookup_by_name_and_location(self, name: str, city: str = None, 
                                   state: str = None, version: str = "2.1") -> Optional[List[Dict]]:
        """
        Look up provider by name and location
        
        Args:
            name: Provider name
            city: City name
            state: State code
            version: API version
            
        Returns:
            List of matching providers or None
        """
        self._rate_limit()
        
        params = {
            "version": version
        }
        
        # Determine if searching for organization or individual
        if "," in name or any(word in name.upper() for word in ["LLC", "INC", "CORP", "LLP", "HOSPITAL", "CLINIC"]):
            params["organization_name"] = name
        else:
            # Try to parse as individual name
            name_parts = name.strip().split()
            if len(name_parts) >= 2:
                params["first_name"] = name_parts[0]
                params["last_name"] = " ".join(name_parts[1:])
        
        if city:
            params["city"] = city
        if state:
            params["state"] = state
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("result_count", 0) > 0:
                return data["results"]
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching provider {name}: {e}")
            return None


class ProviderIdentityVerifier:
    """Main class for verifying provider identity and specialty"""
    
    def __init__(self, db_path: str = "claims_database.db"):
        """
        Initialize verifier
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.api_client = NPPESAPIClient()
    
    def get_unique_providers(self) -> List[Dict]:
        """
        Extract unique providers from claims database
        
        Returns:
            List of unique provider records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
        SELECT DISTINCT
            Provider_NPI,
            Service_Provider_Name,
            Service_Provider_City,
            Service_Provider_State,
            Service_Provider_Zip,
            Provider_Taxonomy_Code,
            Group_Name,
            Provider_ID
        FROM ClaimsData
        WHERE Provider_NPI IS NOT NULL AND Provider_NPI != ''
        """
        
        cursor.execute(query)
        providers = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return providers
    
    def verify_provider(self, npi: str, name: str = None, city: str = None, 
                       state: str = None, claims_taxonomy: str = None) -> ProviderIdentity:
        """
        Verify provider identity using NPPES API
        
        Args:
            npi: National Provider Identifier
            name: Provider name from claims
            city: City from claims
            state: State from claims
            claims_taxonomy: Taxonomy code from claims
            
        Returns:
            ProviderIdentity object with verification results
        """
        verification_reasons = []
        
        # Try NPI lookup first
        npi_data = self.api_client.lookup_npi(npi)
        
        if not npi_data:
            verification_reasons.append(f"NPI {npi} not found in NPPES registry")
            return ProviderIdentity(
                npi=npi,
                provider_type="Unknown",
                name=name or "Unknown",
                identity_established=False,
                verification_reasons=verification_reasons
            )
        
        # Parse NPI data
        identity = self._parse_npi_response(npi_data, claims_taxonomy)
        
        # Verify identity matches claims data
        identity_established = True
        
        # Check name match
        if name:
            name_match = self._check_name_match(name, identity)
            if not name_match:
                verification_reasons.append(
                    f"Name mismatch: Claims='{name}' vs NPPES='{identity.name}'"
                )
                identity_established = False
            else:
                verification_reasons.append("Name verified successfully")
        
        # Check location match
        if city and state:
            location_match = self._check_location_match(city, state, identity)
            if not location_match:
                verification_reasons.append(
                    f"Location mismatch: Claims='{city}, {state}' vs NPPES='{identity.city}, {identity.state}'"
                )
                identity_established = False
            else:
                verification_reasons.append("Location verified successfully")
        
        # Check taxonomy match
        if claims_taxonomy and identity.taxonomy_code:
            if claims_taxonomy != identity.taxonomy_code:
                verification_reasons.append(
                    f"Taxonomy mismatch: Claims='{claims_taxonomy}' vs NPPES='{identity.taxonomy_code}'"
                )
            else:
                verification_reasons.append("Taxonomy code verified successfully")
        
        # Check NPI status
        if identity.status and identity.status != "A":
            verification_reasons.append(f"NPI status: {identity.status} (not active)")
            identity_established = False
        else:
            verification_reasons.append("NPI is active")
        
        identity.identity_established = identity_established
        identity.verification_reasons = verification_reasons
        
        return identity
    
    def _parse_npi_response(self, npi_data: Dict, claims_taxonomy: str = None) -> ProviderIdentity:
        """Parse NPPES API response into ProviderIdentity object"""
        
        basic = npi_data.get("basic", {})
        npi = npi_data.get("number")
        
        # Determine provider type
        enumeration_type = npi_data.get("enumeration_type")
        if enumeration_type == "NPI-1":
            provider_type = "Individual"
            first_name = basic.get("first_name", "")
            last_name = basic.get("last_name", "")
            name = f"{first_name} {last_name}".strip()
        else:
            provider_type = "Organization"
            name = basic.get("organization_name", "Unknown")
            first_name = None
            last_name = None
        
        # Get primary taxonomy
        taxonomies = npi_data.get("taxonomies", [])
        primary_taxonomy = None
        taxonomy_code = None
        taxonomy_desc = None
        
        for tax in taxonomies:
            if tax.get("primary"):
                primary_taxonomy = tax
                break
        
        if not primary_taxonomy and taxonomies:
            primary_taxonomy = taxonomies[0]
        
        if primary_taxonomy:
            taxonomy_code = primary_taxonomy.get("code")
            taxonomy_desc = primary_taxonomy.get("desc")
        
        # Use claims taxonomy if NPPES doesn't have one
        if not taxonomy_code and claims_taxonomy:
            taxonomy_code = claims_taxonomy
        
        # Get primary practice address
        addresses = npi_data.get("addresses", [])
        primary_address = None
        
        for addr in addresses:
            if addr.get("address_purpose") == "LOCATION":
                primary_address = addr
                break
        
        if not primary_address and addresses:
            primary_address = addresses[0]
        
        address_line = None
        city = None
        state = None
        zip_code = None
        phone = None
        
        if primary_address:
            address_parts = []
            if primary_address.get("address_1"):
                address_parts.append(primary_address["address_1"])
            if primary_address.get("address_2"):
                address_parts.append(primary_address["address_2"])
            address_line = ", ".join(address_parts) if address_parts else None
            city = primary_address.get("city")
            state = primary_address.get("state")
            zip_code = primary_address.get("postal_code")
            phone = primary_address.get("telephone_number")
        
        return ProviderIdentity(
            npi=npi,
            provider_type=provider_type,
            name=name,
            first_name=first_name,
            last_name=last_name,
            taxonomy_code=taxonomy_code,
            taxonomy_description=taxonomy_desc,
            primary_specialty=taxonomy_desc,
            address=address_line,
            city=city,
            state=state,
            zip_code=zip_code,
            phone=phone,
            enumeration_date=basic.get("enumeration_date"),
            last_updated=basic.get("last_updated"),
            status=basic.get("status")
        )
    
    def _check_name_match(self, claims_name: str, identity: ProviderIdentity) -> bool:
        """Check if names match (fuzzy matching)"""
        if not claims_name:
            return True
        
        claims_name_clean = claims_name.upper().strip()
        identity_name_clean = identity.name.upper().strip()
        
        # Direct match
        if claims_name_clean == identity_name_clean:
            return True
        
        # Partial match (one contains the other)
        if claims_name_clean in identity_name_clean or identity_name_clean in claims_name_clean:
            return True
        
        # For individuals, check last name match
        if identity.provider_type == "Individual" and identity.last_name:
            if identity.last_name.upper() in claims_name_clean:
                return True
        
        return False
    
    def _check_location_match(self, claims_city: str, claims_state: str, 
                             identity: ProviderIdentity) -> bool:
        """Check if location matches"""
        if not claims_city or not claims_state:
            return True
        
        if not identity.city or not identity.state:
            return False
        
        city_match = claims_city.upper().strip() == identity.city.upper().strip()
        state_match = claims_state.upper().strip() == identity.state.upper().strip()
        
        return city_match and state_match
    
    def verify_all_providers(self, output_file: str = "provider_verification_results.json"):
        """
        Verify all providers in the database and save results
        
        Args:
            output_file: Path to output JSON file
        """
        providers = self.get_unique_providers()
        results = []
        
        print(f"Found {len(providers)} unique providers to verify")
        
        for i, provider in enumerate(providers, 1):
            print(f"\nVerifying {i}/{len(providers)}: NPI {provider['Provider_NPI']}")
            
            identity = self.verify_provider(
                npi=provider['Provider_NPI'],
                name=provider['Service_Provider_Name'],
                city=provider['Service_Provider_City'],
                state=provider['Service_Provider_State'],
                claims_taxonomy=provider['Provider_Taxonomy_Code']
            )
            
            result = {
                "claims_data": provider,
                "nppes_data": asdict(identity),
                "verification_timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Print summary
            status = "✓ VERIFIED" if identity.identity_established else "✗ NOT VERIFIED"
            print(f"  Status: {status}")
            print(f"  Type: {identity.provider_type}")
            print(f"  Name: {identity.name}")
            print(f"  Specialty: {identity.primary_specialty or 'N/A'}")
            for reason in identity.verification_reasons:
                print(f"    - {reason}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nVerification complete. Results saved to {output_file}")
        
        # Print summary statistics
        verified_count = sum(1 for r in results if r['nppes_data']['identity_established'])
        print(f"\nSummary:")
        print(f"  Total providers: {len(results)}")
        print(f"  Verified: {verified_count}")
        print(f"  Not verified: {len(results) - verified_count}")
        print(f"  Verification rate: {verified_count/len(results)*100:.1f}%")


def main():
    """Main execution function"""
    
    # Initialize verifier
    verifier = ProviderIdentityVerifier("claims_database.db")
    
    # Option 1: Verify all providers
    print("Starting provider verification process...")
    verifier.verify_all_providers("provider_verification_results.json")
    
    # Option 2: Verify single provider (example)
    # identity = verifier.verify_provider(
    #     npi="1234567893",
    #     name="John Smith",
    #     city="New York",
    #     state="NY"
    # )
    # print(json.dumps(asdict(identity), indent=2))


if __name__ == "__main__":
    main()