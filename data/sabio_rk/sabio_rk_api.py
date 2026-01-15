#!/usr/bin/env python3
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
    print(f"\nGlycolysis entries: {count_entries('Pathway:glycolysis')}")

    # Example: Count human enzyme entries
    print(f"Human enzyme entries: {count_entries('Organism:Homo sapiens')}")

    # Example: Get a specific entry
    print(f"\nFetching entry 14792...")
    sbml = get_kinetic_entry(14792)
    print(f"Retrieved {len(sbml)} bytes of SBML data")
