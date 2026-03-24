# utils/reward_utils.py
import os
import sys
import math
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem
from MDAnalysis import Universe

from .settings import settings
from .system_prep import SystemPrep


def plif_recovery(
    mol_pred_sdf: str,
    mol_true_sdf: str,
    receptor_pdb: str,
    ir_cache: Optional[dict] = None,   # per-epoch cache
) -> float:
    """
    Compute proportion of ground-truth interactions recovered by the predicted pose,
    using the same SystemPrep pipeline and settings from your ProLIF analysis.

    Identical behavior to the non-cached version, except GT prep/counts are cached
    per (mol_true_sdf, receptor_pdb).
    """
    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

    # --- Match original early failure behavior for TRUE ligand load ---
    try:
        _lig_true_probe = Chem.MolFromMolFile(mol_true_sdf, sanitize=True)
    except Exception:
        print(f"[plif_recovery] failed to load EVEN THE TRUE ligand from {mol_true_sdf}")
        return 0.00

    if ir_cache is None:
        ir_cache = {}

    key = (mol_true_sdf, receptor_pdb)
    entry = ir_cache.get(key)
    if entry is None:
        print(f'building cache for {key}')
        # Build and store GT prep once
        prep = SystemPrep(sanitize=False, optimize_hydrogens=False)

        # Prepare TRUE ligand and POCKET (receptor) once
        lig_true_for_prep, pocket = prep.prepare(mol_true_sdf, receptor_pdb)

        # Compute GT counts once (mirror original logic and robustness)
        fp_true = plf.Fingerprint(
            interactions=settings.interactions,
            parameters=settings.interaction_parameters,
            count=True,
        )
        fp_true.run_from_iterable([lig_true_for_prep], pocket, progress=False)
        xtal_ifp = fp_true.ifp.get(0, {})

        if not xtal_ifp:
            # Same as original: treat as no GT interactions
            entry = {
                "prep": prep,
                "pocket": pocket,
                "state": "no_gt",          # will print & return 1.0
                "xtal_df": None,
                "xtal_counts": {},
                "total_xtal": 0,
                "df_error_msg": None,
            }
        else:
            print(f'using cached GT prep for {key}')
            try:
                xtal_df = plf.to_dataframe({0: xtal_ifp}, settings.interactions, count=True)
                xtal_counts = xtal_df.droplevel("ligand", axis=1).iloc[0].to_dict()
                total_xtal = sum(xtal_counts.values())
                entry = {
                    "prep": prep,
                    "pocket": pocket,
                    "state": "ok",
                    "xtal_df": xtal_df,       # kept so we can print exactly like original
                    "xtal_counts": xtal_counts,
                    "total_xtal": total_xtal,
                    "df_error_msg": None,
                }
            except (KeyError, IndexError, ValueError) as e:
                # Same as original: treat as 1.0 with the same print message
                entry = {
                    "prep": prep,
                    "pocket": pocket,
                    "state": "df_error",      # will print & return 1.0
                    "xtal_df": None,
                    "xtal_counts": {},
                    "total_xtal": 0,
                    "df_error_msg": str(e),
                }

        ir_cache[key] = entry

    # Unpack cached entry
    prep         = entry["prep"]
    pocket       = entry["pocket"]
    state        = entry["state"]
    xtal_df      = entry["xtal_df"]
    xtal_counts  = entry["xtal_counts"]
    total_xtal   = entry["total_xtal"]
    df_error_msg = entry["df_error_msg"]

    # --- Early exits that exactly mirror original behavior/prints ---
    if state == "no_gt":
        print(f'nothnig in xtal_ifp')
        return 1.0

    if state == "df_error":
        print(f"[plif_recovery] No interactions in true ligand ({df_error_msg}); returning 1.0")
        return 1.0

    # From here on, identical to original logic

    # Step 1: Load molecules (PRED + TRUE template for bond orders)
    try:
        lig_pred = Chem.MolFromMolFile(mol_pred_sdf, sanitize=False)
    except Exception:
        print(f"[plif_recovery] failed to load predicted ligand from {mol_pred_sdf}")
        return 0.00

    # Reload TRUE as template (original does this before assign)
    lig_true_tpl = Chem.MolFromMolFile(mol_true_sdf, sanitize=True)
    if lig_true_tpl is None:
        print(f"[plif_recovery] failed to load EVEN THE TRUE ligand from {mol_true_sdf}")
        return 0.00

    # Step 2: Remove hydrogens from lig_pred
    lig_pred = Chem.RemoveHs(lig_pred)

    # Step 3: Kekulize the template (true ligand)
    Chem.Kekulize(lig_true_tpl, clearAromaticFlags=True)

    # Step 4: Assign bond orders from template
    lig_fixed = AllChem.AssignBondOrdersFromTemplate(lig_true_tpl, lig_pred)

    # Step 5: Manually kekulize the fixed molecule
    Chem.Kekulize(lig_fixed, clearAromaticFlags=True)

    # Step 6: Save to a temporary file
    fd, fixed_path = tempfile.mkstemp(suffix='.sdf')
    os.close(fd)
    w = Chem.SDWriter(fixed_path)
    w.write(lig_fixed)
    w.close()

    # Step 7: Prepare ligands using your ProLIF SystemPrep class (reusing cached pocket/prep)
    try:
        lig_pred_prepared, _ = prep.prepare(fixed_path, receptor_pdb)
    except Exception as e:
        print(f"[plif_recovery] predicted pocket prep failed ({e}); returning zero recovery")
        os.remove(fixed_path)
        return 0.00
    os.remove(fixed_path)

    # Step 8: Run PLIF recovery (print TRUE first to match original)
    #print("\n=== TRUE LIGAND FINGERPRINT ===")
    #print(xtal_df)

    fp = plf.Fingerprint(
        interactions=settings.interactions,
        parameters=settings.interaction_parameters,
        count=True,
    )
    fp.run_from_iterable([lig_pred_prepared], pocket, progress=False)
    pred_ifp = fp.ifp.get(0, {})
    try:
        pred_df = plf.to_dataframe({0: pred_ifp}, settings.interactions, count=True)
        pred_counts = pred_df.droplevel("ligand", axis=1).iloc[0].to_dict()
    except (KeyError, IndexError, ValueError) as e:
        print(f"[plif_recovery] No interactions in predicted ligand ({e}); returning 0.0")
        return 0.0

    #print("\n=== PREDICTED LIGAND FINGERPRINT ===")
    #print(pred_df)

    recovered = sum(min(xtal_counts[it], pred_counts.get(it, 0)) for it in xtal_counts)
    return (recovered / total_xtal) if total_xtal > 0 else 0.0

