#!/usr/bin/env python
"""
Download real enzyme kinetics datasets for TACTIC evaluation.

Datasets:
1. EnzymeML SLAC-ABTS (Laccase) - DaRUS DOI: 10.18419/darus-2096
2. FAIRDOMHub PGK (Phosphoglycerate kinase) - ID: 4238
3. FAIRDOMHub PFK (Phosphofructokinase) - ID: 4501
"""

import requests
import zipfile
import io
from pathlib import Path
import sys


def download_enzymeml_slac(output_dir: Path) -> bool:
    """
    EnzymeML SLAC-ABTS dataset from DaRUS.
    DOI: 10.18419/darus-2096

    Contains: Laccase kinetics at 5 temperatures (25, 30, 35, 40, 45°C)
    Mechanism: Michaelis-Menten irreversible (known)
    """
    output_dir = output_dir / "enzymeml" / "slac"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading EnzymeML SLAC-ABTS dataset...")
    print("="*60)
    print(f"Source: DaRUS DOI 10.18419/darus-2096")
    print(f"Output: {output_dir}")

    # DaRUS API endpoint
    url = "https://darus.uni-stuttgart.de/api/access/dataset/:persistentId?persistentId=doi:10.18419/darus-2096"

    try:
        response = requests.get(url, allow_redirects=True, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')

        if 'zip' in content_type or response.content[:4] == b'PK\x03\x04':
            # It's a zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(output_dir)
            print(f"  Extracted {len(z.namelist())} files")
            return True
        else:
            # Save raw response for inspection
            with open(output_dir / "raw_response.txt", 'wb') as f:
                f.write(response.content)
            print(f"  Saved raw response for inspection")
            print(f"  Content-Type: {content_type}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        print(f"\n  Manual download required:")
        print(f"  URL: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2096")
        print(f"  Save to: {output_dir}")
        return False


def download_fairdomhub_file(file_id: int, output_path: Path, description: str) -> bool:
    """
    Download a file from FAIRDOMHub.
    """
    url = f"https://fairdomhub.org/data_files/{file_id}/download"

    print(f"\nDownloading {description}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, allow_redirects=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"  Downloaded {len(response.content)} bytes")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        print(f"\n  Manual download required:")
        print(f"  URL: https://fairdomhub.org/data_files/{file_id}")
        return False


def download_fairdomhub_pgk(output_dir: Path) -> bool:
    """
    SulfoSys PGK kinetics (v2) from FAIRDOMHub.
    ID: 4238

    Contains: Phosphoglycerate kinase with substrate variation
    Mechanism: Ordered bi-bi (known)
    """
    print("\n" + "="*60)
    print("Downloading FAIRDOMHub PGK dataset...")
    print("="*60)

    return download_fairdomhub_file(
        file_id=4238,
        output_path=output_dir / "fairdomhub" / "pgk_kinetics.xlsx",
        description="PGK kinetics (SulfoSys)"
    )


def download_fairdomhub_pfk(output_dir: Path) -> bool:
    """
    PFK-1 and PFK-2 kinetics from FAIRDOMHub.
    ID: 4501

    Contains: Phosphofructokinase kinetics
    Mechanism: Allosteric / substrate inhibition (known)
    """
    print("\n" + "="*60)
    print("Downloading FAIRDOMHub PFK dataset...")
    print("="*60)

    return download_fairdomhub_file(
        file_id=4501,
        output_path=output_dir / "fairdomhub" / "pfk_kinetics.xlsx",
        description="PFK kinetics"
    )


def list_downloaded_files(data_dir: Path):
    """List all downloaded files."""
    print("\n" + "="*60)
    print("Downloaded Files:")
    print("="*60)

    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            rel_path = path.relative_to(data_dir)
            print(f"  {rel_path} ({size:,} bytes)")


def main():
    data_dir = Path(__file__).parent.parent / "data" / "real"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TACTIC Real Data Downloader")
    print("="*60)
    print(f"Output directory: {data_dir}")

    results = {}

    # Download each dataset
    results['SLAC'] = download_enzymeml_slac(data_dir)
    results['PGK'] = download_fairdomhub_pgk(data_dir)
    results['PFK'] = download_fairdomhub_pfk(data_dir)

    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    for name, success in results.items():
        status = "OK" if success else "FAILED (manual download required)"
        print(f"  {name}: {status}")

    list_downloaded_files(data_dir)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
