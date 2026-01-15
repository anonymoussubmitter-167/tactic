#!/usr/bin/env python3
"""
TACTIC-Kinetics Data Download Script

This script downloads all required datasets for the TACTIC-Kinetics project.
Run with: python scripts/download_all_data.py
"""

import os
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional

# Base directories
DATA_DIR = Path(__file__).parent.parent / "data"
EQUILIBRATOR_DIR = DATA_DIR / "equilibrator"
ENZYMEML_DIR = DATA_DIR / "enzymeml"
FAIRDOMHUB_DIR = DATA_DIR / "fairdomhub"
SABIO_RK_DIR = DATA_DIR / "sabio_rk"
BRENDA_DIR = DATA_DIR / "brenda"


def download_file(url: str, dest_path: Path, description: str, expected_md5: Optional[str] = None) -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        if expected_md5:
            with open(dest_path, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 == expected_md5:
                print(f"  Already exists with correct checksum, skipping.")
                return True
        else:
            print(f"  Already exists, skipping. Delete to re-download.")
            return True

    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  Downloaded successfully ({dest_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_equilibrator_data():
    """Download eQuilibrator thermodynamic data from Zenodo."""
    print("\n" + "="*60)
    print("EQUILIBRATOR / COMPONENT CONTRIBUTION DATA")
    print("="*60)

    # Component Contribution Training Data (Zenodo 5495826)
    training_files = [
        ("https://zenodo.org/records/5495826/files/TECRDB.csv?download=1",
         "TECRDB.csv", "NIST TECR thermodynamic data"),
        ("https://zenodo.org/records/5495826/files/formation_energies_transformed.csv?download=1",
         "formation_energies_transformed.csv", "Standard formation energies"),
        ("https://zenodo.org/records/5495826/files/redox.csv?download=1",
         "redox.csv", "Redox potential data"),
    ]

    for url, filename, desc in training_files:
        download_file(url, EQUILIBRATOR_DIR / filename, desc)

    # Note: The compound database is 1.3GB and is downloaded automatically
    # by equilibrator-api on first use
    print("\nNote: The equilibrator compound database (1.3GB) will be downloaded")
    print("automatically when you first use the equilibrator-api package.")
    print("To pre-download, run: python -c \"from equilibrator_api import ComponentContribution; ComponentContribution()\"")


def download_enzymeml_slac_data():
    """Download EnzymeML SLAC-ABTS laccase dataset from DaRUS."""
    print("\n" + "="*60)
    print("ENZYMEML SLAC-ABTS LACCASE DATASET")
    print("="*60)

    base_url = "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/"

    # Raw absorption data
    raw_data = [
        ("5", "SLAC_25C.txt", "Raw absorption data 25°C"),
        ("3", "SLAC_30C.txt", "Raw absorption data 30°C"),
        ("1", "SLAC_35C.txt", "Raw absorption data 35.5°C"),
        ("6", "SLAC_40C.txt", "Raw absorption data 40°C"),
        ("2", "SLAC_45C.txt", "Raw absorption data 45°C"),
    ]

    for file_id, filename, desc in raw_data:
        download_file(base_url + file_id, ENZYMEML_DIR / "raw" / filename, desc)

    # EnzymeML documents (OMEX archives)
    omex_files = [
        ("12", "SLAC_25C.omex", "EnzymeML document 25°C"),
        ("9", "SLAC_30C.omex", "EnzymeML document 30°C"),
        ("15", "SLAC_35C.omex", "EnzymeML document 35.5°C"),
        ("7", "SLAC_40C.omex", "EnzymeML document 40°C"),
        ("4", "SLAC_45C.omex", "EnzymeML document 45°C"),
    ]

    for file_id, filename, desc in omex_files:
        download_file(base_url + file_id, ENZYMEML_DIR / "omex" / filename, desc)

    # Calibration data
    calibration_files = [
        ("13", "ABTS_standard_25C.json", "Calibration data 25°C"),
        ("11", "ABTS_standard_30C.json", "Calibration data 30°C"),
        ("14", "ABTS_standard_35C.json", "Calibration data 35.5°C"),
        ("10", "ABTS_standard_40C.json", "Calibration data 40°C"),
        ("8", "ABTS_standard_45C.json", "Calibration data 45°C"),
    ]

    for file_id, filename, desc in calibration_files:
        download_file(base_url + file_id, ENZYMEML_DIR / "calibration" / filename, desc)

    # Analysis notebook
    download_file(base_url + "17", ENZYMEML_DIR / "slac_kinetics.ipynb", "Analysis notebook")
    download_file(base_url + "16", ENZYMEML_DIR / "requirements.txt", "Python requirements")


def download_fairdomhub_data():
    """Download FAIRDOMHub PGK and PFK kinetic datasets."""
    print("\n" + "="*60)
    print("FAIRDOMHUB KINETIC DATASETS")
    print("="*60)

    print("\nNote: FAIRDOMHub datasets require license agreement.")
    print("Please contact the data creators before using for research.")

    # PGK Kinetic Data
    download_file(
        "https://fairdomhub.org/data_files/1148/content_blobs/1761/download",
        FAIRDOMHUB_DIR / "PGK_Kinetics.xls",
        "PGK Kinetic Data (SulfoSys)"
    )

    # PFK Kinetic Data
    download_file(
        "https://fairdomhub.org/data_files/1150/content_blobs/1764/download",
        FAIRDOMHUB_DIR / "PFK_Kinetics.xls",
        "PFK Kinetic Data"
    )


def create_sabio_rk_examples():
    """Create example scripts for SABIO-RK API access."""
    print("\n" + "="*60)
    print("SABIO-RK API EXAMPLES")
    print("="*60)

    example_script = '''#!/usr/bin/env python3
"""
SABIO-RK API Example Scripts

These examples demonstrate how to access kinetic data from SABIO-RK.
"""

import requests
from typing import Optional

BASE_URL = "https://sabiork.h-its.org/sabioRestWebServices"


def check_api_status() -> bool:
    """Check if SABIO-RK API is available."""
    response = requests.get(f"{BASE_URL}/status")
    return response.text.strip() == "UP"


def get_kinetic_entry(entry_id: int) -> str:
    """Get a single kinetic entry by ID (returns SBML)."""
    url = f"{BASE_URL}/kineticLaws/{entry_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def search_kinetics(query: str, format: str = "sbml") -> str:
    """
    Search for kinetic entries.

    Query examples:
    - "Organism:Homo sapiens"
    - "ECNumber:1.1.1.1"
    - "Pathway:glycolysis"
    - "Tissue:liver AND Organism:Homo sapiens"

    Formats: sbml, biopax
    """
    url = f"{BASE_URL}/searchKineticLaws/{format}"
    response = requests.get(url, params={"q": query})
    response.raise_for_status()
    return response.text


def count_entries(query: str) -> int:
    """Count kinetic entries matching a query."""
    url = f"{BASE_URL}/searchKineticLaws/count"
    response = requests.get(url, params={"q": query, "format": "txt"})
    response.raise_for_status()
    return int(response.text.strip())


def get_available_fields() -> str:
    """Get list of available query fields."""
    url = f"{BASE_URL}/searchKineticLaws"
    response = requests.get(url, params={"format": "xml"})
    response.raise_for_status()
    return response.text


def download_enzyme_kinetics(ec_number: str, output_file: str):
    """Download all kinetic data for an EC number."""
    query = f"ECNumber:{ec_number}"
    count = count_entries(query)
    print(f"Found {count} entries for EC {ec_number}")

    if count > 0:
        sbml_data = search_kinetics(query)
        with open(output_file, 'w') as f:
            f.write(sbml_data)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    print("SABIO-RK API Status:", "UP" if check_api_status() else "DOWN")

    # Example: Count glycolysis entries
    print(f"\\nGlycolysis entries: {count_entries('Pathway:glycolysis')}")

    # Example: Count human enzyme entries
    print(f"Human enzyme entries: {count_entries('Organism:Homo sapiens')}")

    # Example: Get a specific entry
    print(f"\\nFetching entry 14792...")
    sbml = get_kinetic_entry(14792)
    print(f"Retrieved {len(sbml)} bytes of SBML data")
'''

    SABIO_RK_DIR.mkdir(parents=True, exist_ok=True)
    script_path = SABIO_RK_DIR / "sabio_rk_api.py"
    with open(script_path, 'w') as f:
        f.write(example_script)
    print(f"Created SABIO-RK API example script: {script_path}")


def create_brenda_examples():
    """Create example scripts for BRENDA API access."""
    print("\n" + "="*60)
    print("BRENDA API EXAMPLES")
    print("="*60)

    example_script = '''#!/usr/bin/env python3
"""
BRENDA SOAP API Example Scripts

These examples demonstrate how to access kinetic data from BRENDA.
NOTE: You need to register at https://www.brenda-enzymes.org/ to use the API.
"""

import hashlib
from typing import Optional

# You need to install zeep: pip install zeep
try:
    from zeep import Client, Settings
    ZEEP_AVAILABLE = True
except ImportError:
    ZEEP_AVAILABLE = False
    print("Warning: zeep not installed. Run: pip install zeep")


WSDL_URL = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"


def get_password_hash(password: str) -> str:
    """Hash password using SHA-256 as required by BRENDA."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_client():
    """Create BRENDA SOAP client."""
    if not ZEEP_AVAILABLE:
        raise ImportError("zeep package required. Install with: pip install zeep")
    settings = Settings(strict=False)
    return Client(WSDL_URL, settings=settings)


def get_km_values(email: str, password: str, ec_number: str, organism: Optional[str] = None):
    """
    Get Km values for an enzyme.

    Args:
        email: Your registered BRENDA email
        password: Your BRENDA password (will be hashed)
        ec_number: EC number (e.g., "1.1.1.1")
        organism: Optional organism filter (e.g., "Homo sapiens")
    """
    client = create_client()
    password_hash = get_password_hash(password)

    params = [
        email, password_hash,
        f"ecNumber*{ec_number}",
        f"organism*{organism}" if organism else "organism*",
        "kmValue*", "kmValueMaximum*",
        "substrate*", "commentary*",
        "ligandStructureId*", "literature*"
    ]

    result = client.service.getKmValue(*params)
    return result


def get_turnover_numbers(email: str, password: str, ec_number: str, organism: Optional[str] = None):
    """Get kcat (turnover number) values for an enzyme."""
    client = create_client()
    password_hash = get_password_hash(password)

    params = [
        email, password_hash,
        f"ecNumber*{ec_number}",
        f"organism*{organism}" if organism else "organism*",
        "turnoverNumber*", "turnoverNumberMaximum*",
        "substrate*", "commentary*", "literature*"
    ]

    result = client.service.getTurnoverNumber(*params)
    return result


def get_ki_values(email: str, password: str, ec_number: str, organism: Optional[str] = None):
    """Get Ki (inhibition constant) values for an enzyme."""
    client = create_client()
    password_hash = get_password_hash(password)

    params = [
        email, password_hash,
        f"ecNumber*{ec_number}",
        f"organism*{organism}" if organism else "organism*",
        "kiValue*", "kiValueMaximum*",
        "inhibitor*", "commentary*", "literature*"
    ]

    result = client.service.getKiValue(*params)
    return result


def parse_brenda_result(result_string: str) -> list:
    """
    Parse BRENDA result string into list of dictionaries.

    BRENDA returns data as: "field1*value1#field2*value2!field1*value1#..."
    where # separates fields and ! separates entries.
    """
    if not result_string:
        return []

    entries = result_string.split("!")
    parsed = []

    for entry in entries:
        if not entry.strip():
            continue
        fields = entry.split("#")
        entry_dict = {}
        for field in fields:
            if "*" in field:
                key, value = field.split("*", 1)
                entry_dict[key] = value
        if entry_dict:
            parsed.append(entry_dict)

    return parsed


if __name__ == "__main__":
    print("BRENDA API Example")
    print("-" * 40)
    print("To use the BRENDA API, you need to:")
    print("1. Register at https://www.brenda-enzymes.org/")
    print("2. Set your email and password in this script")
    print("3. Run the example functions")
    print()
    print("Example usage:")
    print("  km_data = get_km_values('your@email.com', 'password', '1.1.1.1', 'Homo sapiens')")
    print("  parsed = parse_brenda_result(km_data)")
'''

    BRENDA_DIR.mkdir(parents=True, exist_ok=True)
    script_path = BRENDA_DIR / "brenda_api.py"
    with open(script_path, 'w') as f:
        f.write(example_script)
    print(f"Created BRENDA API example script: {script_path}")


def main():
    """Main download function."""
    print("TACTIC-Kinetics Data Download Script")
    print("="*60)

    # Create directories
    for dir_path in [EQUILIBRATOR_DIR, ENZYMEML_DIR, FAIRDOMHUB_DIR, SABIO_RK_DIR, BRENDA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Download/create each dataset
    download_equilibrator_data()
    download_enzymeml_slac_data()
    download_fairdomhub_data()
    create_sabio_rk_examples()
    create_brenda_examples()

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nData directory structure:")
    print(f"  {DATA_DIR}/")
    print(f"    equilibrator/    - Thermodynamic training data")
    print(f"    enzymeml/        - SLAC laccase kinetics")
    print(f"    fairdomhub/      - PGK and PFK kinetics")
    print(f"    sabio_rk/        - SABIO-RK API scripts")
    print(f"    brenda/          - BRENDA API scripts")
    print("\nNext steps:")
    print("1. Install Python dependencies: pip install equilibrator-api zeep requests")
    print("2. For BRENDA access, register at https://www.brenda-enzymes.org/")
    print("3. For FAIRDOMHub data, contact the data creators for license")


if __name__ == "__main__":
    main()
