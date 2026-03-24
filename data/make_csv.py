"""
make_csv.py

Scan a base directory with structure:
    {base_dir}/{cid}/{cid}_protein.pdb
    {base_dir}/{cid}/{cid}_ligand.sdf

and write a CSV suitable for passing to inference.py --protein_ligand_csv.

Only includes complexes whose IDs are listed in posebusters_308_ids.txt
(located in the same directory as this script).

Usage:
    python make_csv.py --base_dir /path/to/dataset
    python make_csv.py --base_dir /path/to/dataset --output my_input.csv
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing {cid}/{cid}_protein.pdb and {cid}/{cid}_ligand.sdf')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_df.csv'),
                        help='Output CSV path (default: inference_df.csv next to this script)')
    args = parser.parse_args()

    rows = []
    base_dir = os.path.abspath(args.base_dir)

    # Load allowed IDs from posebusters_308_ids.txt (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ids_path = os.path.join(script_dir, 'posebusters_308_ids.txt')

    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f'Could not find ID list at {ids_path}')

    with open(ids_path, 'r') as f:
        allowed_ids = set(line.strip() for line in f if line.strip())

    print(f'Loaded {len(allowed_ids)} allowed IDs from {ids_path}')

    found_ids = set()

    for cid in sorted(os.listdir(base_dir)):
        if cid not in allowed_ids:
            continue

        cid_dir = os.path.join(base_dir, cid)
        if not os.path.isdir(cid_dir):
            continue

        protein = os.path.join(cid_dir, f'{cid}_protein.pdb')
        ligand = os.path.join(cid_dir, f'{cid}_ligand.sdf')

        if not os.path.isfile(protein):
            print(f'  [SKIP] {cid}: missing {os.path.basename(protein)}')
            continue
        if not os.path.isfile(ligand):
            print(f'  [SKIP] {cid}: missing {os.path.basename(ligand)}')
            continue

        rows.append({
            'complex_name': cid,
            'experimental_protein': protein,
            'ligand': ligand
        })

        found_ids.add(cid)

    # Report missing IDs (useful sanity check)
    missing_ids = allowed_ids - found_ids
    if missing_ids:
        print(f'Warning: {len(missing_ids)} IDs from list not found in dataset')

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['complex_name', 'experimental_protein', 'ligand'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written {len(rows)} complexes to {args.output}')


if __name__ == '__main__':
    main()