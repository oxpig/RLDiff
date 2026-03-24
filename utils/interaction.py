import os
import sys
import math
from dataclasses import dataclass, field

import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem
from MDAnalysis import Universe
import tempfile, os
from uuid import uuid4

from .settings import settings
from .system_prep import SystemPrep


def plif_recovery(
    mol_pred_sdf: str,
    mol_true_sdf: str,
    receptor_pdb: str,
) -> float:
    """
    Compute proportion of ground-truth interactions recovered by the predicted pose,
    using the same SystemPrep pipeline and settings from your ProLIF analysis.
    """
    import tempfile
    from rdkit.Chem import rdmolops
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

    prep = SystemPrep(sanitize=False, optimize_hydrogens=False)

    # Step 1: Load molecules
    try:
        lig_true = Chem.MolFromMolFile(mol_true_sdf, sanitize=True)
    except:
        print(f"[plif_recovery] failed to load EVEN THE TRUE ligand from {mol_true_sdf}")
        return 0.00
    try:
        lig_pred = Chem.MolFromMolFile(mol_pred_sdf, sanitize=False)
    except:
        print(f"[plif_recovery] failed to load predicted ligand from {mol_pred_sdf}")
        return 0.00

    # Step 2: Remove hydrogens from lig_pred
    lig_pred = Chem.RemoveHs(lig_pred)

    # Step 3: Kekulize the template
    Chem.Kekulize(lig_true, clearAromaticFlags=True)

    # Step 4: Assign bond orders from template
    lig_fixed = AllChem.AssignBondOrdersFromTemplate(lig_true, lig_pred)

    # Step 5: Manually kekulize the fixed molecule
    Chem.Kekulize(lig_fixed, clearAromaticFlags=True)

    # Step 6: Save to a fixed file path
    tmpdir = "/data/dragon317"
    if not (os.path.isdir(tmpdir) and os.access(tmpdir, os.W_OK)):
        tmpdir = tempfile.gettempdir()

    fixed_path = os.path.join(tmpdir, f"tmp_ligand_{os.getpid()}_{uuid4().hex}.sdf")
    w = Chem.SDWriter(fixed_path)
    w.write(lig_fixed)
    w.close()


    # Step 7: Prepare ligands using your ProLIF SystemPrep class
    lig_true, pocket = prep.prepare(mol_true_sdf, receptor_pdb)
    try:
        lig_pred, _ = prep.prepare(fixed_path, receptor_pdb)
    except Exception as e:
        print(f"[plif_recovery] predicted pocket prep failed ({e}); returning zero recovery")
        os.remove(fixed_path)
        return 0.00

    os.remove(fixed_path)

    # Step 8: Run PLIF recovery
    fp = plf.Fingerprint(
        interactions=settings.interactions,
        parameters=settings.interaction_parameters,
        count=True,
    )

    fp.run_from_iterable([lig_true], pocket, progress=False)
    xtal_ifp = fp.ifp.get(0, {})
    if not xtal_ifp:
        print(f'nothnig in xtal_ifp')
        return 1.0


    try:
        xtal_df = plf.to_dataframe({0: xtal_ifp}, settings.interactions, count=True)
        xtal_counts = xtal_df.droplevel("ligand", axis=1).iloc[0].to_dict()
        total_xtal = sum(xtal_counts.values())
    except (KeyError, IndexError, ValueError) as e:
        print(f"[plif_recovery] No interactions in true ligand ({e}); returning 1.0")
        return 1.0


    fp.ifp.clear()
    fp.run_from_iterable([lig_pred], pocket, progress=False)
    pred_ifp = fp.ifp.get(0, {})
    try:
        pred_df = plf.to_dataframe({0: pred_ifp}, settings.interactions, count=True)
        pred_counts = pred_df.droplevel("ligand", axis=1).iloc[0].to_dict()
    except (KeyError, IndexError, ValueError) as e:
        print(f"[plif_recovery] No interactions in predicted ligand ({e}); returning 0.0")
        return 0.0
        


    recovered = sum(
        min(xtal_counts[it], pred_counts.get(it, 0))
        for it in xtal_counts
    )

    return (recovered / total_xtal) if total_xtal > 0 else 0.0

