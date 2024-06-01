from rdkit import Chem
import mols2grid
from src.al_loop import LeadCompound
from src.sas_score import compute_ertl_score


def show_molecules(molecules):
    if isinstance(molecules[0], Chem.Mol):
        return mols2grid.display(molecules)
    elif isinstance(molecules[0], LeadCompound):
        mols = []
        subset = []
        include_activity = False
        include_sa = False
        for mol in molecules:
            m = Chem.MolFromSmiles(mol.smiles)
            if mol.activity:
                m.SetProp('Activity', f'Activity: {mol.activity:.4f}')
                include_activity = True
            if mol.synth_score:
                m.SetProp('Synthesizability', f'Synthesizability: {mol.synth_score:.4f}')
                include_sa = True
            else:
                m.SetProp('Synthesizability', f'Synthesizability: ???')
            mols.append(m)
        if include_activity:
            subset.append('Activity')
        if include_sa:
            subset.append('Synthesizability')
        return mols2grid.display(mols, subset=subset)


def evaluate_synthesizability(molecules):
    for mol in molecules:
        mol.synth_score = compute_ertl_score(mol.smiles)
