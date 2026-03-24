"""
minimize_utils.py

Post-inference pipeline: merge per-rank SDF outputs, minimize with smina (Vina scoring),
then rerank minimized poses with GNINA.

Adapted from merge_sdfs.py and minimize_poses_gnina_rerank.py.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

# Folder name written by inference.py: index{idx}___{complex_name}
_FOLDER_RX = re.compile(r"^index\d+___(.+)$")
# SDF file name: rank{n}.sdf or rank{n}_confidence{conf}.sdf
_RANK_RX = re.compile(r"^rank(?P<rank>\d+)(?:_confidence(?P<conf>[-\d.]+))?\.sdf$")


# ─────────────────────────── merge ────────────────────────────────────

def _collect_sdfs(out_dir: Path, complex_keys=None) -> Dict[str, List[Tuple[int, float, Path]]]:
    """
    Scan out_dir for index*___<complex_key> folders and collect rank SDF files.
    Returns {complex_key: [(rank, confidence, sdf_path), ...]} sorted by rank.
    If complex_keys is provided, only include matching keys.
    """
    complexes: Dict[str, List[Tuple[int, float, Path]]] = {}

    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = _FOLDER_RX.match(sub.name)
        if not m:
            continue
        complex_key = m.group(1)
        if complex_keys is not None and complex_key not in complex_keys:
            continue

        per_rank: Dict[int, Tuple[float, Path]] = {}
        for sdf in sub.glob("*.sdf"):
            rm = _RANK_RX.match(sdf.name)
            if not rm:
                continue
            rank = int(rm.group("rank"))
            conf = float(rm.group("conf")) if rm.group("conf") else 0.0
            # Prefer the file with confidence in the name
            if rank not in per_rank or (rm.group("conf") and "_confidence" not in per_rank[rank][1].name):
                per_rank[rank] = (conf, sdf)

        poses = [(rank, conf, path) for rank, (conf, path) in sorted(per_rank.items())]
        if poses:
            complexes[complex_key] = poses

    return complexes


def merge_sdfs(out_dir: Path, complex_keys=None) -> Path:
    """
    Merge per-rank SDF files from inference output into one SDF per complex.
    Returns the merged_sdfs/ directory path.
    """
    merged_dir = out_dir / "merged_sdfs"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    complexes = _collect_sdfs(out_dir, complex_keys=complex_keys)
    if not complexes:
        print(f"[merge] No SDF files found in {out_dir}")
        return merged_dir

    total_mols = 0
    for complex_key, poses in sorted(complexes.items()):
        out_sdf = merged_dir / f"{complex_key}.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        n_written = 0
        for rank, conf, sdf_path in poses:
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
            for mol in suppl:
                if mol is None:
                    print(f"  [WARN] Could not read mol from {sdf_path.name}, skipping")
                    continue
                mol.SetProp("rank", str(rank))
                mol.SetProp("confidence", f"{conf:.4f}")
                mol.SetProp("original_filename", sdf_path.name)
                mol.SetProp("complex_id", complex_key)
                writer.write(mol)
                n_written += 1
        writer.close()
        total_mols += n_written
        print(f"  [merge] {complex_key}: {len(poses)} files -> {n_written} molecules")

    print(f"[merge] Done: {len(complexes)} complexes, {total_mols} total molecules -> {merged_dir}")
    return merged_dir


# ─────────────────────────── smina / gnina helpers ────────────────────

def _run_smina_score_only(protein_pdb: Path, ligands_sdf: Path, output_sdf: Path,
                          smina_path: str, env: Dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [smina_path, "-r", str(protein_pdb), "-l", str(ligands_sdf),
         "--scoring", "vina", "--cpu", "1", "--score_only", "-o", str(output_sdf)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )


def _run_smina_minimize(protein_pdb: Path, ligands_sdf: Path, output_sdf: Path,
                        smina_path: str, env: Dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [smina_path, "-r", str(protein_pdb), "-l", str(ligands_sdf),
         "--scoring", "vina", "--cpu", "1", "--minimize", "-o", str(output_sdf)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )


def _run_gnina_rerank(protein_pdb: Path, ligands_sdf: Path, output_sdf: Path,
                      gnina_path: str, env: Dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [gnina_path, "-r", str(protein_pdb), "-l", str(ligands_sdf),
         "--score_only", "--cpu", "1", "-o", str(output_sdf)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )


def _read_sdf_property(sdf_path: Path, prop_name: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for mol in suppl:
        if mol is not None and mol.HasProp(prop_name):
            try:
                values.append(float(mol.GetProp(prop_name)))
            except (ValueError, TypeError):
                values.append(None)
        else:
            values.append(None)
    return values


def _read_first_available_property(sdf_path: Path,
                                    prop_names: List[str]) -> Tuple[Optional[str], List[Optional[float]]]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    n_mols = sum(1 for mol in suppl if mol is not None)
    for prop in prop_names:
        vals = _read_sdf_property(sdf_path, prop)
        if any(v is not None for v in vals):
            return prop, vals
    return None, [None] * n_mols


# ─────────────────────────── per-complex worker ───────────────────────

def _minimize_one(merged_sdf: Path, output_sdf: Path, protein_pdb: Path,
                  complex_key: str, smina_path: str, gnina_path: str
                  ) -> Tuple[str, float, float, float, int, Optional[str]]:
    """
    1) Vina pre-score with smina
    2) Vina minimize with smina
    3) GNINA rerank minimized poses
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Step 1: Vina pre-score
    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            prescored_sdf = Path(tmp.name)
        proc = _run_smina_score_only(protein_pdb, merged_sdf, prescored_sdf, smina_path, env)
        t_score = time.time() - t0
        if proc.returncode != 0:
            prescored_sdf.unlink(missing_ok=True)
            shutil.copyfile(merged_sdf, output_sdf)
            return (complex_key, t_score, 0.0, 0.0, proc.returncode,
                    f"smina score_only failed:\n{proc.stdout}")
        prescored_sdf.unlink(missing_ok=True)
    except Exception as e:
        t_score = time.time() - t0
        shutil.copyfile(merged_sdf, output_sdf)
        return (complex_key, t_score, 0.0, 0.0, -1, f"pre-score exception: {e}\n{traceback.format_exc()}")

    # Step 2: Vina minimize
    t1 = time.time()
    try:
        proc = _run_smina_minimize(protein_pdb, merged_sdf, output_sdf, smina_path, env)
        t_min = time.time() - t1
        if proc.returncode != 0:
            shutil.copyfile(merged_sdf, output_sdf)
            return (complex_key, t_score, t_min, 0.0, proc.returncode,
                    f"smina minimize failed:\n{proc.stdout}")
    except Exception as e:
        t_min = time.time() - t1
        shutil.copyfile(merged_sdf, output_sdf)
        return (complex_key, t_score, t_min, 0.0, -1, f"minimize exception: {e}\n{traceback.format_exc()}")

    # Step 3: GNINA rerank
    t2 = time.time()
    try:
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            gnina_sdf = Path(tmp.name)
        proc = _run_gnina_rerank(protein_pdb, output_sdf, gnina_sdf, gnina_path, env)
        t_rerank = time.time() - t2
        if proc.returncode != 0:
            gnina_sdf.unlink(missing_ok=True)
            return (complex_key, t_score, t_min, t_rerank, proc.returncode,
                    f"gnina rerank failed:\n{proc.stdout}")

        # Collect scored poses, sort by CNNscore descending, write rank{N}_gnina{score}.sdf
        per_rank_dir = output_sdf.parent / complex_key
        per_rank_dir.mkdir(parents=True, exist_ok=True)
        suppl = Chem.SDMolSupplier(str(gnina_sdf), removeHs=False, sanitize=False)
        scored = []
        for mol in suppl:
            if mol is None:
                continue
            try:
                score_val = float(mol.GetProp("CNNscore")) if mol.HasProp("CNNscore") else 0.0
            except (ValueError, TypeError):
                score_val = 0.0
            scored.append((score_val, mol))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (score_val, mol) in enumerate(scored, start=1):
            rank_sdf = per_rank_dir / f"rank{rank}_gnina{score_val:.2f}.sdf"
            writer = Chem.SDWriter(str(rank_sdf))
            writer.write(mol)
            writer.close()

        shutil.move(str(gnina_sdf), str(output_sdf))
    except Exception as e:
        t_rerank = time.time() - t2
        return (complex_key, t_score, t_min, t_rerank, -1,
                f"rerank exception: {e}\n{traceback.format_exc()}")

    return (complex_key, t_score, t_min, t_rerank, 0, None)


def _worker(task: Tuple) -> Tuple:
    merged_sdf, output_sdf, protein_pdb, complex_key, smina_path, gnina_path = task
    return _minimize_one(merged_sdf, output_sdf, protein_pdb, complex_key, smina_path, gnina_path)


# ─────────────────────────── public entry point ───────────────────────

def minimize_and_rerank(out_dir: Path,
                        protein_path_map: Dict[str, str],
                        smina_path: str = "smina",
                        gnina_path: str = "gnina",
                        n_workers: int = 4) -> None:
    """
    Merge inference outputs then minimize + rerank.

    Args:
        out_dir:          --out_dir from inference (contains index*___* folders)
        protein_path_map: {complex_key: protein_pdb_path} where complex_key matches
                          the folder suffix after index\d+___ (with "/" replaced by "-")
        smina_path:       path/name of smina executable
        gnina_path:       path/name of gnina executable
        n_workers:        parallel workers for minimize/rerank
    """
    out_dir = Path(out_dir)
    complex_keys = set(protein_path_map.keys())

    # Step 1: merge
    print("\n[minimize] Step 1/2: merging SDF outputs...")
    merged_dir = merge_sdfs(out_dir, complex_keys=complex_keys)

    # Step 2: minimize + rerank
    print("\n[minimize] Step 2/2: smina minimize + gnina rerank...")
    minimized_dir = out_dir / "minimized_poses"
    if minimized_dir.exists():
        shutil.rmtree(minimized_dir)
    minimized_dir.mkdir(parents=True)

    tasks = []
    for sdf in sorted(merged_dir.glob("*.sdf")):
        complex_key = sdf.stem
        protein_path = protein_path_map.get(complex_key)
        if protein_path is None:
            print(f"  [WARN] No protein path for '{complex_key}', skipping")
            continue
        protein_pdb = Path(protein_path)
        if not protein_pdb.is_file():
            print(f"  [WARN] Protein PDB not found: {protein_pdb}, skipping")
            continue
        output_sdf = minimized_dir / sdf.name
        tasks.append((sdf, output_sdf, protein_pdb, complex_key, smina_path, gnina_path))

    if not tasks:
        print("[minimize] No complexes to process.")
        return

    print(f"  Processing {len(tasks)} complexes with {n_workers} workers...")
    results = []
    t_wall = time.time()

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_map = {executor.submit(_worker, t): t[3] for t in tasks}
            for fut in as_completed(future_map):
                cid = future_map[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    _, t_score, t_min, t_rerank, rc, err = result
                    if rc == 0:
                        print(f"    [OK]   {cid}: score={t_score:.1f}s min={t_min:.1f}s rerank={t_rerank:.1f}s", flush=True)
                    else:
                        print(f"    [FAIL] {cid}: rc={rc} | {str(err)[:120]}", flush=True)
                except Exception as e:
                    print(f"    [ERROR] {cid}: {e}", flush=True)
                    results.append((cid, 0.0, 0.0, 0.0, -1, str(e)))
    else:
        for task in tasks:
            result = _worker(task)
            results.append(result)
            _, t_score, t_min, t_rerank, rc, err = result
            cid = task[3]
            if rc == 0:
                print(f"    [OK]   {cid}: score={t_score:.1f}s min={t_min:.1f}s rerank={t_rerank:.1f}s", flush=True)
            else:
                print(f"    [FAIL] {cid}: rc={rc} | {str(err)[:120]}", flush=True)

    t_wall = time.time() - t_wall
    n_ok = sum(1 for r in results if r[4] == 0)
    n_fail = len(results) - n_ok

    print(f"\n[minimize] Done: {n_ok} OK, {n_fail} failed in {t_wall:.1f}s")
    print(f"  Outputs: {minimized_dir}")
