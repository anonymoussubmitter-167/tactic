# TACTIC-Kinetics Data Sources

This document describes all data sources required for the TACTIC-Kinetics project and how to access them.

---

## 1. Thermodynamic Grounding Data

### 1.1 eQuilibrator / Component Contribution

**Description**: Standard Gibbs energies for ~500,000 metabolic compounds and ~10,000 reactions.

**Use Cases**:
- Set priors on ΔG°_rxn
- Validate predicted energies
- Provide supervision signal for thermodynamic consistency loss

#### Data Sources on Zenodo

| Dataset | DOI | Size | Description |
|---------|-----|------|-------------|
| Compound Database | [10.5281/zenodo.4128543](https://zenodo.org/records/4128543) | 1.3 GB | SQLite database of ~500k compounds from MetaNetX v3.1 |
| Component Contribution Training Data | [10.5281/zenodo.5495826](https://zenodo.org/records/5495826) | 1.7 MB | Measured ΔG values from NIST TECR database |

#### Training Data Files (Zenodo 5495826)

| File | Format | Size | Content |
|------|--------|------|---------|
| `TECRDB.csv` | CSV | 1.7 MB | Thermodynamic data from NIST TECR |
| `formation_energies_transformed.csv` | CSV | 51.3 KB | Standard formation energies |
| `redox.csv` | CSV | 8.9 KB | Redox potential data |

#### Python API Installation

```bash
pip install equilibrator-api
```

#### Python Usage Example

```python
from equilibrator_api import ComponentContribution, Q_

# Initialize (downloads ~1.3GB database on first run)
cc = ComponentContribution()

# Calculate ΔG for a reaction
reaction = cc.parse_reaction_formula("kegg:C00002 + kegg:C00001 = kegg:C00008 + kegg:C00009")
dG_prime = cc.standard_dg_prime(reaction)
print(f"ΔG'° = {dG_prime}")
```

#### Direct Download Links

```bash
# Compound database (1.3 GB)
wget https://zenodo.org/records/4128543/files/compounds.sqlite

# Training data
wget https://zenodo.org/records/5495826/files/TECRDB.csv
wget https://zenodo.org/records/5495826/files/formation_energies_transformed.csv
wget https://zenodo.org/records/5495826/files/redox.csv
```

---

### 1.2 BRENDA (BRaunschweig ENzyme DAtabase)

**Description**: Comprehensive enzyme functional data including kinetic parameters (Km, kcat, Ki, IC50).

**Use Cases**:
- Derive empirical distributions of activation energies
- Validate kinetic predictions
- Build priors for enzyme kinetic parameters

#### Access Methods

**Option 1: SOAP API (Recommended for automated access)**

Registration required at: https://www.brenda-enzymes.org/

```python
from zeep import Client, Settings
import hashlib

wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
password = hashlib.sha256("your_password".encode("utf-8")).hexdigest()
settings = Settings(strict=False)
client = Client(wsdl, settings=settings)

# Example: Get Km values for EC 1.1.1.1 in Homo sapiens
parameters = (
    "your_email@example.com", password,
    "ecNumber*1.1.1.1",
    "organism*Homo sapiens",
    "kmValue*", "kmValueMaximum*",
    "substrate*", "commentary*",
    "ligandStructureId*", "literature*"
)
result = client.service.getKmValue(*parameters)
```

**Available SOAP Methods for Kinetic Parameters:**

| Method | Description |
|--------|-------------|
| `getKmValue()` | Michaelis constant (Km) |
| `getTurnoverNumber()` | Turnover number (kcat) |
| `getKiValue()` | Inhibition constant (Ki) |
| `getIc50Value()` | IC50 values |
| `getKcatKmValue()` | Catalytic efficiency (kcat/Km) |

**Option 2: Python Parsers for Flat File**

```bash
# brendapy (recommended)
pip install brendapy

# BRENDApyrser (alternative)
pip install brendapyrser
```

Download flat file from BRENDA website (requires license agreement).

---

### 1.3 SABIO-RK (Biochemical Reaction Kinetics Database)

**Description**: Curated database of ~71,000 kinetic entries from >7,300 publications.

**Use Cases**:
- Retrieve kinetic rate equations and parameters
- Access experimental conditions (pH, temperature, buffer)
- Export data in SBML format for modeling

#### REST API Endpoints

Base URL: `https://sabiork.h-its.org/sabioRestWebServices`

| Endpoint | Description | Example |
|----------|-------------|---------|
| `/status` | API health check | `/status` |
| `/kineticLaws/{ID}` | Single entry by ID | `/kineticLaws/14792` |
| `/kineticLaws?kinlawids=X,Y` | Multiple entries | `/kineticLaws?kinlawids=123,234` |
| `/searchKineticLaws/sbml?q=` | Search (SBML output) | `?q=Organism:"Homo sapiens"` |
| `/searchKineticLaws/count?q=` | Count matching entries | `?q=ECNumber:1.1.1.1` |
| `/reactions/reactionIDs?q=` | Get reaction IDs | `?q=Pathway:"glycolysis"` |

#### Query Examples

```bash
# Get kinetic entry 14792 as SBML
curl "https://sabiork.h-its.org/sabioRestWebServices/kineticLaws/14792"

# Search for human liver enzymes
curl "https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/sbml?q=Tissue:liver%20AND%20Organism:Homo%20sapiens"

# Count glycolysis entries
curl "https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/count?q=Pathway:glycolysis&format=txt"

# Get available search fields
curl "https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws?format=xml"
```

#### Python Usage

```python
import requests

def get_kinetic_entry(entry_id):
    url = f"https://sabiork.h-its.org/sabioRestWebServices/kineticLaws/{entry_id}"
    response = requests.get(url)
    return response.text  # Returns SBML

def search_kinetics(query):
    url = "https://sabiork.h-its.org/sabioRestWebServices/searchKineticLaws/sbml"
    response = requests.get(url, params={"q": query})
    return response.text
```

---

## 2. Real Kinetic Datasets (Benchmarking)

### 2.1 EnzymeML SLAC-ABTS (DaRUS)

**Description**: Laccase (SLAC) catalyzed oxidation of ABTS across temperatures 25-45°C.

**DOI**: [10.18419/darus-3867](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3867)

**License**: CC BY 4.0

#### Files Available

| File | Format | Size | Description |
|------|--------|------|-------------|
| Raw absorption data (5 files) | .txt | ~25 KB each | Measurements at 25, 30, 35, 40, 45°C |
| Calibration data (5 files) | .json | ~5 KB each | ABTS standards per temperature |
| EnzymeML documents (5 files) | .omex | ~13 KB each | Full experimental data + fitted parameters |
| `slac_kinetics.ipynb` | .ipynb | 3.6 MB | Analysis notebook |
| `requirements.txt` | .txt | 223 B | Python dependencies |

#### Download Commands

```bash
# Raw data files
wget -O data/enzymeml/SLAC_25C.txt "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/5"
wget -O data/enzymeml/SLAC_30C.txt "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/3"
wget -O data/enzymeml/SLAC_35C.txt "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/1"
wget -O data/enzymeml/SLAC_40C.txt "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/6"
wget -O data/enzymeml/SLAC_45C.txt "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/2"

# EnzymeML documents
wget -O data/enzymeml/SLAC_25C.omex "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/12"
wget -O data/enzymeml/SLAC_30C.omex "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/9"
wget -O data/enzymeml/SLAC_35C.omex "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/15"
wget -O data/enzymeml/SLAC_40C.omex "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/7"
wget -O data/enzymeml/SLAC_45C.omex "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/4"

# Calibration data
wget -O data/enzymeml/ABTS_standard_25C.json "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/13"
wget -O data/enzymeml/ABTS_standard_30C.json "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/11"
wget -O data/enzymeml/ABTS_standard_35C.json "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/14"
wget -O data/enzymeml/ABTS_standard_40C.json "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/10"
wget -O data/enzymeml/ABTS_standard_45C.json "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/8"

# Analysis notebook
wget -O data/enzymeml/slac_kinetics.ipynb "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/DARUS-3867/17"
```

#### Alternative: GitHub Repository

```bash
git clone https://github.com/EnzymeML/slac_modeling.git data/enzymeml/slac_modeling
```

---

### 2.2 SulfoSys PGK (FAIRDOMHub)

**Description**: Phosphoglycerate kinase kinetic characterization from *Sulfolobus solfataricus*.

**URL**: https://fairdomhub.org/assays/222

**Project**: SulfoSys - Central Carbon Metabolism of Sulfolobus solfataricus

#### Files Available

| File | Format | Size | URL |
|------|--------|------|-----|
| PGK_Kinetics-SEEK.xls | Excel | 114 KB | https://fairdomhub.org/data_files/1148 |

#### Download

```bash
wget -O data/fairdomhub/PGK_Kinetics.xls "https://fairdomhub.org/data_files/1148/content_blobs/1761/download"
```

#### Important Notes

- **License**: No default license - contact creators for permission
- **Creators**: Dawie van Niekerk, Jacky Snoep (University of Stellenbosch)
- **Experimental conditions**: ATP, ADP, BPG, 3PG concentration variations
- **Temperature studies**: 30°C and 70°C comparisons available

---

### 2.3 PFK Kinetics (FAIRDOMHub)

**Description**: Phosphofructokinase kinetic characterization from *Plasmodium falciparum*.

**URL**: https://fairdomhub.org/data_files/1150

**Project**: Glucose metabolism in Plasmodium falciparum

#### Files Available

| File | Format | Size | URL |
|------|--------|------|-----|
| PFK_kinetics-SEEK.xls | Excel | 138 KB | https://fairdomhub.org/data_files/1150 |

#### Download

```bash
wget -O data/fairdomhub/PFK_Kinetics.xls "https://fairdomhub.org/data_files/1150/content_blobs/1764/download"
```

#### Important Notes

- **License**: No default license - contact creators for permission
- **Creators**: Dawie van Niekerk, Jacky Snoep
- **Associated research**: Whole body modelling of glucose metabolism in malaria patients

---

## 3. Summary Table

| Data Source | Type | Size | Access | License |
|-------------|------|------|--------|---------|
| eQuilibrator Compounds | Thermodynamic | 1.3 GB | Python API / Zenodo | CC BY 4.0 |
| Component Contribution Training | Thermodynamic | 1.7 MB | Zenodo | CC BY 4.0 |
| BRENDA | Kinetic Parameters | Variable | SOAP API | License required |
| SABIO-RK | Kinetic Parameters | Variable | REST API | Free (commercial license) |
| EnzymeML SLAC-ABTS | Progress Curves | ~4 MB | DaRUS / GitHub | CC BY 4.0 |
| SulfoSys PGK | Kinetic Data | 114 KB | FAIRDOMHub | Contact required |
| PFK Kinetics | Kinetic Data | 138 KB | FAIRDOMHub | Contact required |

---

## 4. References

1. Noor, E., et al. "eQuilibrator 3.0: a database solution for thermodynamic constant estimation." *Nucleic Acids Research* 50.D1 (2022): D603-D609.

2. Chang, A., et al. "BRENDA, the ELIXIR core data resource in 2021: new developments and updates." *Nucleic Acids Research* 49.D1 (2021): D498-D508.

3. Wittig, U., et al. "SABIO-RK: an updated resource for manually curated biochemical reaction kinetics." *Nucleic Acids Research* 46.D1 (2018): D656-D660.

4. Range, J., et al. "EnzymeML—a data exchange format for biocatalysis and enzymology." *The FEBS Journal* 289.19 (2022): 5864-5874.
