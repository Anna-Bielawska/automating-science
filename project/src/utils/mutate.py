import logging

import numpy as np
import selfies
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from selfies import decoder

logger = logging.getLogger(__name__)


def get_ECFP4(mol: object) -> object:
    """
    Return rdkit ECFP4 fingerprint object for mol

    Args:
        mol (rdkit.Chem.rdchem.Mol): RdKit mol object

    Returns:
        rdkit.Chem.rdchem._DataStructs.ExplicitBitVect: ECFP4 fingerprint object
    """
    return AllChem.GetMorganFingerprint(mol, 2)


def get_selfie_chars(selfie: str) -> list[str]:
    """
    Obtain a list of all selfie characters in string selfie

    Args:
        selfie (str) : A selfie string - representing a molecule

    Returns:
        chars_selfie: list of selfie characters present in molecule selfie

    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return chars_selfie


def sanitize_smiles(smi: str) -> tuple:
    """Return a canonical smile representation of smi

    Args:
        smi (str): A SMILES string

    Returns:
        mol (rdkit.Chem.rdchem.Mol): RdKit mol object
        smi_canon (str): Canonical SMILES representation of mol
        True (bool): True if SMILES is valid, False otherwise
    """

    mol = smi2mol(smi, sanitize=True)
    smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
    return (mol, smi_canon, True)


def mutate_selfie(selfie: str, max_molecules_len: int, write_fail_cases: bool = False):
    """Return a mutated selfie string (only one mutation on slefie is performed)

    Mutations are done until a valid molecule is obtained
    Rules of mutation: With a 50% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another

    Args:
        selfie (str): A SELFIE string
        max_molecules_len (int): Maximum length of molecules
        write_fail_cases (bool): Write failed cases to a file

    Returns:
        tuple: A tuple of mutated selfie and canonical SMILES representation of the molecule
    """
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    alphabet = sorted(
        list(selfies.get_semantic_robust_alphabet())
    )  # 34 SELFIE characters

    while not valid:
        fail_counter += 1

        choice_ls = [1, 2]  # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = (
                chars_selfie[:random_index]
                + [random_character]
                + chars_selfie[random_index:]
            )

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[
                    random_index + 1 :
                ]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1 :]
                )

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index] + chars_selfie[random_index + 1 :]
                )

        else:
            raise Exception("Invalid Operation trying to be performed")

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)

        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon == "":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except Exception as e:
            logger.error(f"Failed to decode SELFIE: {e}")
            valid = False
            if fail_counter > 1 and write_fail_cases is True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write(
                    "Tried to mutate SELFIE: "
                    + str(sf)
                    + " To Obtain: "
                    + str(selfie_mutated)
                    + "\n"
                )
                f.close()

    return (selfie_mutated, smiles_canon)
