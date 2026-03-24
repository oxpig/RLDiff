import os
import pandas as pd
from torch_geometric.loader import DataLoader
from datasets.pdbbind import PDBBind
from argparse import Namespace
import tempfile
import numpy as np


def set_nones(l):
    return [s if str(s) != 'nan' else None for s in l]


def read_split_file(filepath):
    """
    Reads the split file and returns a list of complex names.
    """
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def construct_dataframe_from_split(dataset_root, complex_names, mode):
    """
    Constructs a DataFrame containing necessary information for the given complex names.

    Output columns (required by PDBBind):
      - 'complex_name'           (string)
      - 'experimental_protein'   (path to the PDB)
      - 'ligand_path'            (path to the ligand SDF)
      - 'protein_sequence'       (string or None)
      - 'pocket_center'          (torch tensor or None)
    """
    data = {
        "complex_name": [],
        "experimental_protein": [],
        "ligand_path": [],
        "protein_sequence": [],
        "pocket_center": []   # ← we must append to this every time below
    }

    for complex_name in complex_names:
        dataset_root_ = dataset_root
        complex_dir = os.path.join(dataset_root_, complex_name)
        ligand_path = os.path.join(complex_dir, f"{complex_name}_ligand.sdf")
        protein_path = os.path.join(complex_dir, f"{complex_name}_protein.pdb")
        protein_sequence_path = os.path.join(complex_dir, f"{complex_name}_protein_sequence.fasta")

        if os.path.isdir(complex_dir) and os.path.exists(ligand_path) and os.path.exists(protein_path):
            data["complex_name"].append(complex_name)
            data["experimental_protein"].append(protein_path)

            if os.path.exists(protein_sequence_path):
                with open(protein_sequence_path, "r") as seq_file:
                    sequence = "".join([line.strip() for line in seq_file if not line.startswith(">")])
                data["protein_sequence"].append(sequence)
            else:
                data["protein_sequence"].append(None)

            data["ligand_path"].append(ligand_path)
            data["pocket_center"].append(None)
        else:
            print(f"Warning: Missing files for complex '{complex_name}'. Skipping.")

    return pd.DataFrame(data)


# A tiny wrapper so we can feed a list of HeteroData objects into DataLoader.
class GraphListDataset:
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def construct_loaders_with_splits(dataset_root, split_paths, args, score_model_args):
    """
    Constructs train, validation, and test DataLoaders by instantiating PDBBind
    (all_atoms=True, flexible_sidechains=False).

    Args:
        dataset_root (str): Path to the root dataset directory (e.g., PDBBind/).
        split_paths (dict): Paths to the split files {'train': ..., 'val': ..., 'test': ...}.
        args (Namespace): Additional configuration arguments.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1) Read split files
    train_complexes = read_split_file(split_paths["train"])
    val_complexes   = read_split_file(split_paths["val"])
    test_complexes  = read_split_file(split_paths["test"]) if split_paths.get("test") else []

    # 2) Build three DataFrames
    train_df = construct_dataframe_from_split(dataset_root, train_complexes, mode="Train")
    print(train_df)
    val_df   = construct_dataframe_from_split(dataset_root, val_complexes,   mode="Val")
    test_df  = construct_dataframe_from_split(dataset_root, test_complexes,  mode="Test")

    # 3) For each DataFrame, extract parallel lists
    def process_dataframe(df: pd.DataFrame):
        complex_name_list      = set_nones(df["complex_name"].tolist())
        protein_path_list      = set_nones(df["experimental_protein"].tolist())
        ligand_path_list       = set_nones(df["ligand_path"].tolist())
        protein_sequence_list  = set_nones(df["protein_sequence"].tolist())
        pocket_center_list     = df["pocket_center"].tolist()  # now exists

        # Ensure no None in complex_name_list
        complex_name_list = [
            name if name is not None else f"complex_{i}"
            for i, name in enumerate(complex_name_list)
        ]
        return complex_name_list, protein_path_list, protein_sequence_list, ligand_path_list, pocket_center_list

    train_lists = process_dataframe(train_df)
    val_lists   = process_dataframe(val_df)
    test_lists  = process_dataframe(test_df)


    
    def create_pdbbind_dataset(
        complex_names, protein_files, protein_sequences, ligand_paths, pocket_centers,
        split_name="shared"               # you can pass "train"/"val"/"test" if you want separate folders
    ):
        # Re-assemble the DataFrame PDBBind expects
        df = pd.DataFrame({
            "complex_name":      complex_names,
            "experimental_protein": protein_files,
            "ligand_path":       ligand_paths,
            "protein_sequence":  protein_sequences,
            "pocket_center":     pocket_centers,
        })
        print(f"protein_ligand_df:\n{df}")

        # ────────────────────────── cache path (persistent) ──────────────────────
        os.makedirs(args.cache_path, exist_ok=True)                    # ensure parent exists
        dataset_cache_root = os.path.join(args.cache_path, "pdbbind_cache")
        os.makedirs(dataset_cache_root, exist_ok=True)                 # stable root

        dataset_cache = os.path.join(dataset_cache_root, split_name)   # sub-folder per split
        os.makedirs(dataset_cache, exist_ok=True)

        stuff = PDBBind(
            transform=None,
            protein_ligand_df=df,
            chain_cutoff=np.inf,
            receptor_radius=score_model_args.receptor_radius,
            cache_path=dataset_cache,
            remove_hs=score_model_args.remove_hs,
            max_lig_size=None,
            c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
            matching=False,
            keep_original=False,
            conformer_match_sidechains=False,
            use_original_conformer_fallback=True,
            popsize=score_model_args.matching_popsize,
            maxiter=score_model_args.matching_maxiter,
            all_atoms=score_model_args.all_atoms,
            require_ligand=True,
            num_workers=0,
            keep_local_structures=True,
            pocket_reduction=score_model_args.pocket_reduction,
            pocket_buffer=score_model_args.pocket_buffer,
            pocket_cutoff=score_model_args.pocket_cutoff,
            pocket_reduction_mode=score_model_args.pocket_reduction_mode,

            # ✅ force rigid sidechains in DATA
            flexible_sidechains=False,

            # (doesn't matter if these are set when flexible_sidechains=False, but fine to leave)
            flexdist=score_model_args.flexdist,
            flexdist_distance_metric=score_model_args.flexdist_distance_metric,

            fixed_knn_radius_graph=not score_model_args.not_fixed_knn_radius_graph,
            knn_only_graph=not score_model_args.not_knn_only_graph,  # ✅ FIX THIS LINE

            include_miscellaneous_atoms=score_model_args.include_miscellaneous_atoms,
            use_old_wrong_embedding_order=score_model_args.use_old_wrong_embedding_order,
        )



# ── Strip off the “_protein_processed.pdb___…_ligand.sdf” suffix so that `.name` is just the PDBBind ID ──
        for cg in stuff.protein_ligand_df["complex_graph"]:
            original = cg["name"]  # e.g. "6ZPB_3D1_protein_processed.pdb___6ZPB_3D1_ligand.sdf"
            cg["name"] = original.split("_protein")[0]

        # ── Inspect one example’s HeteroData (optional debug) ──────────────────────
        hetero0 = stuff.protein_ligand_df.loc[0, "complex_graph"]
        for node_type in hetero0.node_types:
            for attr_name, tensor in hetero0[node_type].items():
                _ = tuple(tensor.shape) if hasattr(tensor, "shape") else type(tensor)
        for edge_type in hetero0.edge_types:
            for attr_name, tensor in hetero0[edge_type].items():
                _ = tuple(tensor.shape) if hasattr(tensor, "shape") else type(tensor)
        for key, val in hetero0.items():
            if key not in hetero0.node_types and key not in hetero0.edge_types:
                _ = tuple(val.shape) if hasattr(val, "shape") else type(val)

        return stuff


    train_dataset = create_pdbbind_dataset(*train_lists)
    print(f"TRAIN DATASET: {train_dataset}, size = {len(train_dataset)}")
    val_dataset   = create_pdbbind_dataset(*val_lists)
    print(f"VALIDATION DATASET: {val_dataset}, size = {len(val_dataset)}")
    test_dataset  = create_pdbbind_dataset(*test_lists)
    print(f"TEST DATASET: {test_dataset}, size = {len(test_dataset)}")

    # ── Extract all HeteroData graphs from each PDBBind instance ─────────────────
    train_graphs = train_dataset.protein_ligand_df["complex_graph"].tolist()
    val_graphs   = val_dataset.protein_ligand_df["complex_graph"].tolist()
    test_graphs  = test_dataset.protein_ligand_df["complex_graph"].tolist()

    # ── Mark every graph as “success=True” so downstream code can check `g.success[0]` ─────────
    import torch
    for g in train_graphs:
        g["success"] = torch.tensor([True])
    for g in val_graphs:
        g["success"] = torch.tensor([True])
    for g in test_graphs:
        g["success"] = torch.tensor([True])

    # 5) Wrap each list of HeteroData graphs in our GraphListDataset, then DataLoader
    def create_loader_from_graphs(graph_list, shuffle=False):
        dataset = GraphListDataset(graph_list)
        return DataLoader(
            dataset=dataset,
            batch_size=1,  # each iteration yields one HeteroData graph
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=args.dataloader_drop_last if hasattr(args, "dataloader_drop_last") else False,
            #persistent_workers=False,
            #prefetch_factor=0,
            #timeout=3600,
        )

    train_loader = create_loader_from_graphs(train_graphs, shuffle=True)
    val_loader   = create_loader_from_graphs(val_graphs,   shuffle=False)
    test_loader  = create_loader_from_graphs(test_graphs,  shuffle=False)

    return train_loader, val_loader, test_loader

