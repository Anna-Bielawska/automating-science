import os
import sys
from rdkit import Chem

ERTL_SCORE_PATH = None
for path in sys.path:
    if path.endswith("site-packages"):
        ERTL_SCORE_PATH = os.path.join(path, "rdkit/Contrib/SA_Score")
        sys.path.append(ERTL_SCORE_PATH)
        break

# has to be after the previous block because of the sys path update
from src import sascorer


def compute_ertl_score(compound: str):
    mol = Chem.MolFromSmiles(compound)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)