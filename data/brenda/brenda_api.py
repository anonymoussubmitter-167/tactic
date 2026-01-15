#!/usr/bin/env python3
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
