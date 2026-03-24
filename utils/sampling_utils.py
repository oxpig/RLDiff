
import torch
import numpy as np
from typing import Dict, Tuple
from typing import List, Dict
from rdkit import Chem
import os
from argparse import Namespace

def calculate_mean(score: torch.Tensor, g: torch.Tensor, dt: float, current_t: float) -> Dict[str, torch.Tensor]:
    """
    Calculate the mean and pre-scaled mean for a transformation.

    Args:
        score (torch.Tensor): Model scores, shape [batch_size, ...].
        g (torch.Tensor): Scaled sigma, shape [batch_size, 1].
        dt (float): Difference in sigma between current and next timestep.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing 'mean' and 'pre_scaled_mean'.
    """
    pre_scaled_mean_final_prediction = g**2*current_t
    pre_scaled_mean = g**2 * dt  # [batch_size, 1]
    mean = score * pre_scaled_mean  # [batch_size, ...]
    mean_final_pred = score * pre_scaled_mean_final_prediction  # [batch_size, ...]
    return {"mean": mean, "pre_scaled_mean": pre_scaled_mean, "pre_scaled_mean_T": pre_scaled_mean_final_prediction, "mean_final_pred": mean_final_pred}



def sample_noise(
    g: torch.Tensor, 
    dt: float, 
    shape: torch.Size, 
    device: torch.device, 
    add_noise: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample noise for perturbation and calculate z and std_dev.

    Args:
        g (torch.Tensor): Scaled sigma, shape [batch_size, 1].
        dt (float): Difference in sigma between current and next timestep.
        shape (torch.Size): Shape of the noise tensor.
        device (torch.device): Device to sample noise on.
        add_noise (bool): Whether to add noise or not. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - z: Calculated perturbation, shape matching shape.
            - std_dev: Standard deviation, shape [batch_size, 1].
    """
    if add_noise:
        z = torch.normal(mean=0, std=1, size=shape, device=device)
        noise = g * torch.sqrt(torch.tensor(dt, device=g.device)) * z  # [batch_size, ...]
    else:
        z = torch.zeros(shape, device=device)
        noise = z
    
    std_dev = g * torch.sqrt(torch.tensor(dt, device=g.device))   # [batch_size, 1]
    return noise, std_dev, z


def compute_delta_sigma(noise_schedule: list, timestep: int, total_steps: int) -> float:
    """
    Compute the difference in sigma between current and next timestep.

    Args:
        noise_schedule (list): List of sigma values for each timestep.
        timestep (int): Current timestep index.
        total_steps (int): Total number of inference steps.

    Returns:
        float: Delta sigma for the current timestep.
    """
    if timestep < total_steps - 1:
        return noise_schedule[timestep] - noise_schedule[timestep + 1]
    else:
        return noise_schedule[timestep]

def handle_nans(scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handle NaN values in the score tensors by replacing them with small values.

    Args:
        scores (Dict[str, torch.Tensor]): Dictionary of scores for each transformation.
        logger: Logger instance for logging warnings.

    Returns:
        Dict[str, torch.Tensor]: Updated scores with NaNs handled.
    """
    for key, score in scores.items():
        if torch.isnan(score).any():
            num_nans = torch.sum(torch.isnan(score)).item()
            print(f"{key} contains {num_nans} NaNs. Replacing NaNs with small values.")
            scores[key] = score.nan_to_num(nan=0.01 * torch.nanmean(score.abs()), posinf=0.01, neginf=-0.01)
            print(f'we did a handle nans!')
    return scores

def compute_g(sigma: float, sigma_max: float, sigma_min: float) -> torch.Tensor:
    """
    Compute the scaled sigma (g) based on the provided sigma values.

    Args:
        sigma (float): Current sigma value.
        sigma_max (float): Maximum sigma value for the transformation.
        sigma_min (float): Minimum sigma value for the transformation.

    Returns:
        torch.Tensor: Scaled sigma (g), shape [1].
    """
    g = sigma * np.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))

    return g


def prepare_mask_rotate(mask_rotate_attr, device):
    """
    Prepare the mask_rotate attribute by converting it to a tensor on the specified device.

    Args:
        mask_rotate_attr (list, np.ndarray, torch.Tensor): The mask_rotate attribute.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The prepared mask_rotate tensor.
    """
    if isinstance(mask_rotate_attr, torch.Tensor):
        return mask_rotate_attr.to(device)
    elif isinstance(mask_rotate_attr, np.ndarray):
        return torch.from_numpy(mask_rotate_attr).to(device)
    elif isinstance(mask_rotate_attr, list):
        return torch.tensor(mask_rotate_attr).to(device)
    else:
        raise TypeError(f"Unexpected type for mask_rotate: {type(mask_rotate_attr)}")
    
def save_trajectory_as_pdb(sample_idx: int, trajectory: list, save_dir: str):
    """
    Save all intermediate conformers of a trajectory as a single PDB file.

    Args:
        sample_idx (int): Index of the sample.
        trajectory (List[Dict]): List of steps, each containing 'complex_graph' with Chem.Mol.
        save_dir (str): Directory to save PDB files.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    base_mol = None
    for step in trajectory:
        mol = step['complex_graph'].mol[0]
        if base_mol is None:
            # Remove hydrogens to match ligand_pos
            base_mol = Chem.RemoveHs(mol)
            base_mol = Chem.Mol(base_mol)  # Create a copy
            base_mol.RemoveAllConformers()
        else:
            break
    assert base_mol is not None, "No Mol found in trajectory steps."

    # Add conformers from each step
    for idx, step in enumerate(trajectory):
        mol = step['complex_graph'].mol[0]
        conformer = mol.GetConformer()
        base_mol.AddConformer(conformer, assignId=True)

    # Define the PDB filename
    pdb_filename = os.path.join(save_dir, f'sample_{sample_idx}.pdb')

    # Write the multi-conformer Mol to PDB
    Chem.MolToPDBFile(base_mol, pdb_filename)



def slice_torsion_updates(torsion_updates, data):
    """
    Slices torsion updates to return a tensor of size [batch_size, num_torsion_angles].

    Args:
        torsion_updates (torch.Tensor): Tensor of torsion updates of size [batch_size * num_torsion_angles].
        data (dict): Data dictionary containing ligand information.

    Returns:
        torch.Tensor: Tensor of size [batch_size, num_torsion_angles].
    """
    # Determine batch size from data
    batch_size = data.num_graphs

    # Infer the number of torsion angles per graph
    num_torsion_angles = torsion_updates.numel() // batch_size

    # Reshape torsion_updates to [batch_size, num_torsion_angles]
    torsion_updates_per_batch = torsion_updates.view(batch_size, num_torsion_angles)
    
    return torsion_updates_per_batch




def _precompute_cumulative_scales(
    inference_steps: int,
    tr_schedule: List[float],
    rot_schedule: List[float],
    tor_schedule: List[float],
    t_to_sigma,
    model_args: Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    g_tr_vec, g_rot_vec, g_tor_vec = [], [], []
    dt_tr_vec, dt_rot_vec, dt_tor_vec = [], [], []

    for i in range(inference_steps):
        sigma_tr, sigma_rot, sigma_tor, sigma_sc = t_to_sigma(
            tr_schedule[i], rot_schedule[i], tor_schedule[i], tor_schedule[i]
        )

        g_tr_vec.append(
            compute_g(sigma_tr, model_args.tr_sigma_max, model_args.tr_sigma_min).item()
        )
        g_rot_vec.append(
            compute_g(sigma_rot, model_args.rot_sigma_max, model_args.rot_sigma_min).item()
        )
        g_tor_vec.append(
            compute_g(sigma_tor, model_args.tor_sigma_max, model_args.tor_sigma_min).item()
        )

        if i < inference_steps - 1:
            dt_tr_vec.append(tr_schedule[i] - tr_schedule[i + 1])
            dt_rot_vec.append(rot_schedule[i] - rot_schedule[i + 1])
            dt_tor_vec.append(tor_schedule[i] - tor_schedule[i + 1])
        else:
            dt_tr_vec.append(tr_schedule[i])
            dt_rot_vec.append(rot_schedule[i])
            dt_tor_vec.append(tor_schedule[i])

    g_tr_vec, g_rot_vec, g_tor_vec = map(np.array, (g_tr_vec, g_rot_vec, g_tor_vec))
    dt_tr_vec, dt_rot_vec, dt_tor_vec = map(np.array, (dt_tr_vec, dt_rot_vec, dt_tor_vec))

    G_tr_dt  = (g_tr_vec  ** 2) * dt_tr_vec
    G_rot_dt = (g_rot_vec ** 2) * dt_rot_vec
    G_tor_dt = (g_tor_vec ** 2) * dt_tor_vec

    cum_scale_tr  = np.flip(np.cumsum(np.flip(G_tr_dt )))
    cum_scale_rot = np.flip(np.cumsum(np.flip(G_rot_dt)))
    cum_scale_tor = np.flip(np.cumsum(np.flip(G_tor_dt)))

    return cum_scale_tr, cum_scale_rot, cum_scale_tor
