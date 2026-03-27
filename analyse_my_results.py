import os
import re
import json
import math
import argparse
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from posebusters import PoseBusters
from tqdm import tqdm


GNINA_RE = re.compile(r"_gnina(-?\d+(?:\.\d+)?)\.sdf$")
RANK_RE = re.compile(r"rank(\d+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimized_root", type=str, required=True)
    parser.add_argument("--inference_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--top_k_oracle",
        type=int,
        default=None,
        help="If set, oracle is computed over only the top K poses by GNINA. Default: all poses found.",
    )
    return parser.parse_args()


def parse_gnina_score(path: Path):
    m = GNINA_RE.search(path.name)
    return float(m.group(1)) if m else float("-inf")


def parse_rank(path: Path):
    m = RANK_RE.search(path.name)
    return int(m.group(1)) if m else math.inf


def get_pose_files(complex_dir: Path):
    return sorted(
        complex_dir.glob("*.sdf"),
        key=lambda p: (-parse_gnina_score(p), parse_rank(p), p.name),
    )


def is_bool_like_series(s: pd.Series) -> bool:
    vals = s.dropna()
    if len(vals) == 0:
        return False
    return vals.map(lambda x: isinstance(x, bool)).all()


def get_bool_columns(df: pd.DataFrame):
    return [c for c in df.columns if is_bool_like_series(df[c])]


def find_rmsd_le2_col(df: pd.DataFrame):
    preferred = [
        "rmsd_≤_2å",
        "rmsd_<=_2å",
        "rmsd_≤_2a",
        "rmsd_<=_2a",
    ]
    for p in preferred:
        if p in df.columns:
            return p

    candidates = [c for c in df.columns if ("rmsd" in c.lower() and "2" in c.lower())]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates, key=len)[0]

    raise ValueError(f"Could not find RMSD<=2A column. Columns: {list(df.columns)}")


def eval_one_complex(task):
    complex_key, complex_dir_str, protein_path, true_ligand_path = task
    complex_dir = Path(complex_dir_str)
    pose_files = get_pose_files(complex_dir)

    if not pose_files:
        return {
            "complex_name": complex_key,
            "per_pose_rows": [],
            "error": "no sdf poses found",
        }

    buster = PoseBusters(config="redock")
    per_pose_rows = []

    for pose_idx, pose_path in enumerate(pose_files):
        try:
            pb_df = buster.bust(
                mol_pred=str(pose_path),
                mol_true=str(true_ligand_path),
                mol_cond=str(protein_path),
                full_report=False,
            )
            row = pb_df.iloc[0].copy()

            bool_cols = get_bool_columns(pb_df)
            rmsd_col = find_rmsd_le2_col(pb_df)

            pb_valid_cols = [c for c in bool_cols if c != rmsd_col]
            pb_valid = bool(row[pb_valid_cols].all())
            rmsd_le_2 = bool(row[rmsd_col])
            rmsd_le_2_and_pb_valid = bool(rmsd_le_2 and pb_valid)

            out = {
                "complex_name": complex_key,
                "pose_file": str(pose_path),
                "pose_name": pose_path.name,
                "gnina_score": parse_gnina_score(pose_path),
                "original_rank": parse_rank(pose_path),
                "gnina_rank_within_complex": pose_idx + 1,
                "top1_by_gnina": pose_idx == 0,
                "rmsd_le_2": rmsd_le_2,
                "pb_valid": pb_valid,
                "rmsd_le_2_and_pb_valid": rmsd_le_2_and_pb_valid,
                "pb_eval_failed": False,
                "pb_eval_error": "",
            }

            for c in bool_cols:
                out[c] = bool(row[c])

            per_pose_rows.append(out)

        except Exception as e:
            per_pose_rows.append({
                "complex_name": complex_key,
                "pose_file": str(pose_path),
                "pose_name": pose_path.name,
                "gnina_score": parse_gnina_score(pose_path),
                "original_rank": parse_rank(pose_path),
                "gnina_rank_within_complex": pose_idx + 1,
                "top1_by_gnina": pose_idx == 0,
                "rmsd_le_2": False,
                "pb_valid": False,
                "rmsd_le_2_and_pb_valid": False,
                "pb_eval_failed": True,
                "pb_eval_error": str(e),
            })

    return {
        "complex_name": complex_key,
        "per_pose_rows": per_pose_rows,
        "error": "",
    }


def main():
    args = parse_args()

    minimized_root = Path(args.minimized_root)
    inference_csv = Path(args.inference_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pose_csv = out_dir / "pb_eval_per_pose.csv"
    per_complex_csv = out_dir / "pb_eval_per_complex.csv"
    one_line_csv = out_dir / "pb_eval_one_line.csv"
    summary_json = out_dir / "pb_eval_summary.json"

    df = pd.read_csv(inference_csv)

    required_cols = ["complex_name", "experimental_protein", "ligand"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {inference_csv}: {missing}")

    complex_to_paths = {}
    for _, row in df.iterrows():
        complex_key = str(row["complex_name"]).replace("/", "-")
        complex_to_paths[complex_key] = {
            "protein_path": str(row["experimental_protein"]),
            "true_ligand_path": str(row["ligand"]),
        }

    tasks = []
    skipped = []

    for complex_dir in sorted([p for p in minimized_root.iterdir() if p.is_dir()]):
        complex_key = complex_dir.name
        if complex_key not in complex_to_paths:
            skipped.append({"complex_name": complex_key, "reason": "not found in inference_csv"})
            continue

        tasks.append((
            complex_key,
            str(complex_dir),
            complex_to_paths[complex_key]["protein_path"],
            complex_to_paths[complex_key]["true_ligand_path"],
        ))

    print(f"Evaluating {len(tasks)} complexes with {args.num_workers} workers...")

    results = []
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(eval_one_complex, tasks),
            total=len(tasks),
            desc="PoseBusters complexes",
        ):
            results.append(result)

    all_rows = []
    for r in results:
        all_rows.extend(r["per_pose_rows"])

    if not all_rows:
        raise RuntimeError("No poses were evaluated.")

    per_pose_df = pd.DataFrame(all_rows)
    per_pose_df.to_csv(per_pose_csv, index=False)

    per_complex_rows = []
    for complex_name, g in tqdm(
        per_pose_df.groupby("complex_name", sort=True),
        total=per_pose_df["complex_name"].nunique(),
        desc="Aggregating complexes",
    ):
        g = g.sort_values(
            by=["gnina_score", "original_rank", "pose_name"],
            ascending=[False, True, True]
        ).reset_index(drop=True)

        top1 = g.iloc[0]
        oracle_group = g if args.top_k_oracle is None else g.head(args.top_k_oracle)

        per_complex_rows.append({
            "complex_name": complex_name,
            "n_poses": len(g),
            "oracle_pool_size": len(oracle_group),
            "top1_rmsd_leq_2": bool(top1["rmsd_le_2"]),
            "top1_rmsd_leq_2_and_pb_valid": bool(top1["rmsd_le_2_and_pb_valid"]),
            "oracle_rmsd_leq_2": bool(oracle_group["rmsd_le_2"].any()),
            "oracle_rmsd_leq_2_and_pb_valid": bool(oracle_group["rmsd_le_2_and_pb_valid"].any()),
        })

    per_complex_df = pd.DataFrame(per_complex_rows)
    per_complex_df.to_csv(per_complex_csv, index=False)

    n_complexes = len(per_complex_df)
    summary = {
        "n_complexes_evaluated": int(n_complexes),
        "top1_success_rate_rmsd_le_2": float(per_complex_df["top1_rmsd_leq_2"].mean()),
        "top1_success_rate_rmsd_le_2_and_pb_valid": float(per_complex_df["top1_rmsd_leq_2_and_pb_valid"].mean()),
        "oracle_success_rate_rmsd_le_2": float(per_complex_df["oracle_rmsd_leq_2"].mean()),
        "oracle_success_rate_rmsd_le_2_and_pb_valid": float(per_complex_df["oracle_rmsd_leq_2_and_pb_valid"].mean()),
        "n_top1_success_rmsd_le_2": int(per_complex_df["top1_rmsd_leq_2"].sum()),
        "n_top1_success_rmsd_le_2_and_pb_valid": int(per_complex_df["top1_rmsd_leq_2_and_pb_valid"].sum()),
        "n_oracle_success_rmsd_le_2": int(per_complex_df["oracle_rmsd_leq_2"].sum()),
        "n_oracle_success_rmsd_le_2_and_pb_valid": int(per_complex_df["oracle_rmsd_leq_2_and_pb_valid"].sum()),
        "top_k_oracle": args.top_k_oracle,
        "skipped": skipped,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    one_line = pd.DataFrame([{
        "method": "RLDiff+SminaMin+GninaRank",
        "n_complexes": n_complexes,
        "top1_rmsd_le_2": summary["top1_success_rate_rmsd_le_2"],
        "top1_rmsd_le_2_and_pb_valid": summary["top1_success_rate_rmsd_le_2_and_pb_valid"],
        "oracle_rmsd_le_2": summary["oracle_success_rate_rmsd_le_2"],
        "oracle_rmsd_le_2_and_pb_valid": summary["oracle_success_rate_rmsd_le_2_and_pb_valid"],
        "oracle_pool_size": "all" if args.top_k_oracle is None else args.top_k_oracle,
    }])
    one_line.to_csv(one_line_csv, index=False)

    print("\n=== Summary ===")
    print(f"Complexes evaluated: {n_complexes}")
    print(f"Top-1 success (RMSD <= 2A): {summary['top1_success_rate_rmsd_le_2']:.4f}")
    print(f"Top-1 success (RMSD <= 2A and PB-valid): {summary['top1_success_rate_rmsd_le_2_and_pb_valid']:.4f}")
    print(f"Oracle success (RMSD <= 2A): {summary['oracle_success_rate_rmsd_le_2']:.4f}")
    print(f"Oracle success (RMSD <= 2A and PB-valid): {summary['oracle_success_rate_rmsd_le_2_and_pb_valid']:.4f}")
    print(f"\nWrote:\n  {per_pose_csv}\n  {per_complex_csv}\n  {one_line_csv}\n  {summary_json}")


if __name__ == "__main__":
    main()
