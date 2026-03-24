from typing import Literal, Union

import pandas as pd
from posebusters import PoseBusters


def normalise_PB_valid(x: float, floor: float = 0.6, ceil: float = 1.0) -> float:
    """
    Linearly map PB pass rate to [0, 1] with:
      x <= floor  -> 0
      x >= ceil   -> 1
      else        -> (x - floor) / (ceil - floor)

    Examples (floor=0.6, ceil=1.0):
      0.6 -> 0.0
      0.8 -> 0.5
      1.0 -> 1.0
    """
    if ceil <= floor:
        raise ValueError("ceil must be greater than floor")
    if x <= floor:
        return 0.0
    if x >= ceil:
        return 1.0
    return (x - floor) / (ceil - floor)



def posebuster_check(
    mol_pred: str,
    mol_true: str,
    receptor: str,
    mode: Literal["Train", "Val"] = "Val",
    strat: Literal["add", "mult", "rmsd_only"] = "mult",
) -> Union[float, bool]:
    """
    PoseBusters reward with control flow:
      - 'add': fraction of active checks passed
      - 'mult': multiplier version (scale by RMSD pass)
      - 'rmsd_only': return True if RMSD < 2Å, else False
    """
    pb = PoseBusters(config="redock")
    results: pd.DataFrame = pb.bust(
        mol_pred=mol_pred,
        mol_cond=receptor,
        mol_true=mol_true,
        full_report=False,
    )
    row = results.iloc[0].astype(bool)

    if not row.any():
        return False if strat == "rmsd_only" else 0.0

    rmsd_key = next((c for c in row.index if "rmsd" in c.lower()), None)
    rmsd_pass = bool(row[rmsd_key]) if rmsd_key is not None else False

    if strat == "rmsd_only":
        return rmsd_pass

    if mode == "Train":
        if strat == "add":
            return float(row.mean())
        else:  # 'mult'
            other_cols = [c for c in row.index if c != rmsd_key] if rmsd_key else list(row.index)
            other_rate = row[other_cols].mean() if other_cols else 0.0
            #other_rate = normalise_PB_valid(other_rate, floor=0.7, ceil=1.0)
            scale = 1.0 if rmsd_pass else 0.5
            return max(0.0, min(1.0, scale * float(other_rate)))
    else:  # Eval
        return float(row.mean())




def compute_rewards(
    mol_pred: str,
    mol_true: str,
    receptor: str,
    mode: str,
) -> float:
    return posebuster_check(mol_pred, mol_true, receptor, mode=mode, strat="add")

