"""
Usage:

python visualize_compounds.py path_to_json.json path_to_html.html
"""
from typing import List, Dict, Any

import pandas as pd
import mols2grid
import json

def generate_html_molgrid(compounds: List[str], save_file_path: str,
                      properties: Dict[str, List[Any]] = {}):
    """
    Uses mol2grid to save interactive visualization of a set of compounds given as SMILES.
    """
    print(compounds)
    df_columns = {"SMILES": compounds} # "SMILES" is a custom key used by molgrid
    for property_key in properties:
        df_columns[property_key] = properties[property_key]
    return mols2grid.save(pd.DataFrame(df_columns),
                      subset=["img"] + list(properties),
                      size=(200,200),
                      fixedBondLength=60,
                      tooltip=list(properties),output=save_file_path)

if __name__ == "__main__":
    import sys
    generate_html_molgrid(json.load(open(sys.argv[1])), sys.argv[2])