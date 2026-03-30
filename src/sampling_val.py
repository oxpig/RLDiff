import copy
import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Dict, Optional
from argparse import Namespace

from RLDiff.utils.sampling_utils import (
    calculate_mean,
    sample_noise,
    slice_torsion_updates,
    handle_nans,
    compute_g,
    _precompute_cumulative_scales,
    modify_conformer_batch
)

current_dir = os.path.dirname(os.path.abspath(__file__))
diffdock_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(diffdock_dir)
from utils.diffusion_utils import set_time
from utils.visualise import PDBFile


def sampling_val(
    data_list: List[Data],
    model: torch.nn.Module,
    inference_steps: int,
    tr_schedule: List[float],
    rot_schedule: List[float],
    tor_schedule: List[float],
    device: torch.device,
    t_to_sigma,
    model_args: Namespace,
    batch_size: int = 32,
    temp_sampling: List[float] = [1, 1, 1],
    temp_psi: List[float] = [1, 1, 1],
    visualization_list: Optional[List[PDBFile]] = None,
    args=None,
    mode: str = 'Train',
    no_temp: bool = False
) -> List[List[Dict[str, any]]]:
    """
    Perform the sampling process to generate transformation data for ligand poses.
    Incorporates fixed temperature scaling for translation, rotation, and torsion perturbations.
    Always applies noise except in the final timestep and always includes torsion.

    Args:
        data_list (List[Data]): List of data samples.
        model (torch.nn.Module): The trained model for scoring.
        inference_steps (int): Number of inference steps.
        tr_schedule (List[float]): Translation noise schedule.
        rot_schedule (List[float]): Rotation noise schedule.
        tor_schedule (List[float]): Torsion noise schedule.
        device (torch.device): Device to perform computations on.
        t_to_sigma (Callable): Function to convert t to sigma.
        model_args (Namespace): Additional model arguments, should include `out_dir`.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        temp_sampling (List[float], optional): Temperature scaling factors. Defaults to [1, 1, 1].
        temp_psi (List[float], optional): Temperature psi values. Defaults to [1, 1, 1].
        temp_sigma_data (List[float], optional): Temperature sigma data values. Defaults to [1, 1, 1].
        visualization_list (Optional[List[PDBFile]], optional): List to store visualization frames. Defaults to None.

    Returns:
        List[List[Dict[str, any]]]:
            - Trajectories containing transformation data and final ligand positions for each sample.
    """
    # logger = get_logger()
    N = len(data_list)
    trajectories = [[] for _ in range(N)]  # Initialize trajectories for each sample
    translation_score_x = []
    translation_score_y = []
    translation_score_z = []

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    # Prepare mask_rotate
    mask_rotate = torch.from_numpy(data_list[0]['ligand'].mask_rotate[0]).to(device)


    cum_scale_tr, cum_scale_rot, cum_scale_tor = _precompute_cumulative_scales(
        inference_steps, tr_schedule, rot_schedule, tor_schedule, t_to_sigma, model_args
    )

    with torch.no_grad():
        for batch_id, complex_graph_batch in enumerate(loader):
            b = complex_graph_batch.num_graphs
            n = len(complex_graph_batch['ligand'].pos) // b
            complex_graph_batch = complex_graph_batch.to(device)

            for t_idx in range(inference_steps):
                # Retrieve separate schedules
                t_tr, t_rot, t_tor, t_sidechain_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx], tor_schedule[t_idx]

                # Compute delta_sigma for each transformation
                if t_idx < inference_steps - 1:
                    dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1]
                    dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1]
                    dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1]
                else:
                    dt_tr = tr_schedule[t_idx]
                    dt_rot = rot_schedule[t_idx]
                    dt_tor = tor_schedule[t_idx]
                # Convert t to sigma for each transformation
                sigma_tr, sigma_rot, sigma_tor, _ = t_to_sigma(t_tr, t_rot, t_tor, t_sc_tor = t_sidechain_tor)


                # # Apply cropping if needed
                # if hasattr(model_args, 'crop_beyond') and model_args.crop_beyond is not None:
                #     crop_radius_tr = sigma_tr * 3 + model_args.crop_beyond
                #     mod_complex_graph_batch = copy.deepcopy(complex_graph_batch).to_data_list()
                #     for batch in mod_complex_graph_batch:
                #         all_atoms = getattr(model_args, 'all_atoms', False)
                #         crop_beyond(batch, crop_radius_tr, all_atoms)
                #     mod_complex_graph_batch = Batch.from_data_list(mod_complex_graph_batch).to(device)
                # else:
                #     mod_complex_graph_batch = complex_graph_batch
                mod_complex_graph_batch = complex_graph_batch
                # Set timestep information in the batch
                set_time(
                    mod_complex_graph_batch,
                    t=None,
                    t_tr=t_tr,
                    t_rot=t_rot,
                    t_tor=t_tor,
                    t_sidechain_tor=t_tor,
                    batchsize=b,
                    all_atoms=True,
                    asyncronous_noise_schedule=False,
                    device=device,
                    include_miscellaneous_atoms=getattr(model_args, 'include_miscellaneous_atoms', False)
                )

                tr_score, rot_score, tor_score, _ = model(mod_complex_graph_batch)
                if tor_score.numel() == 0:
                    raise ValueError("Torsion score is unexpectedly empty. Skipping this complex.")
                save_mod_complex_graph_batch = mod_complex_graph_batch.clone().cpu()

                # Handle NaNs in scores
                tr_score = handle_nans({'tr': tr_score})['tr']
                rot_score = handle_nans({'rot': rot_score})['rot']
                tor_score = handle_nans({'tor': tor_score})['tor']


                if t_idx < 3:
                    # Assume tr_score has shape (batch_size, 3)
                    tr_score_cpu = tr_score.cpu().tolist()  # e.g. [[x1,y1,z1], [x2,y2,z2], ...]
                    for score in tr_score_cpu:
                        translation_score_x.append(score[0])
                        translation_score_y.append(score[1])
                        translation_score_z.append(score[2])


                # Compute g for all transformations
                tr_g = compute_g(sigma_tr, getattr(model_args, 'tr_sigma_max', 1.0), getattr(model_args, 'tr_sigma_min', 0.1)).to(device)
                rot_g = compute_g(sigma_rot, getattr(model_args, 'rot_sigma_max', 1.0), getattr(model_args, 'rot_sigma_min', 0.1)).to(device)
                tor_g = compute_g(sigma_tor, getattr(model_args, 'tor_sigma_max', 1.0), getattr(model_args, 'tor_sigma_min', 0.1)).to(device)

                mean_tr_dict  = calculate_mean(tr_score,  tr_g,  dt_tr,  t_tr)
                mean_rot_dict = calculate_mean(rot_score, rot_g, dt_rot, t_rot)
                mean_tor_dict = calculate_mean(tor_score, tor_g, dt_tor, t_tor)

                mean_tr,  pre_scaled_mean_tr,  pre_scaled_mean_T_tr  = (
                    mean_tr_dict['mean'],  mean_tr_dict['pre_scaled_mean'],  mean_tr_dict['pre_scaled_mean_T']
                )
                mean_rot, pre_scaled_mean_rot, pre_scaled_mean_T_rot = (
                    mean_rot_dict['mean'], mean_rot_dict['pre_scaled_mean'], mean_rot_dict['pre_scaled_mean_T']
                )
                mean_tor, pre_scaled_mean_tor, pre_scaled_mean_T_tor = (
                    mean_tor_dict['mean'], mean_tor_dict['pre_scaled_mean'], mean_tor_dict['pre_scaled_mean_T']
                )

                cum_scale_tr_step  = cum_scale_tr [t_idx]
                cum_scale_rot_step = cum_scale_rot[t_idx]
                cum_scale_tor_step = cum_scale_tor[t_idx]

                # Determine whether to add noise (always add noise except final timestep)
                is_last_timestep = (t_idx == inference_steps - 1)

                add_noise = not (mode == 'Val' and is_last_timestep)

                # Sample noise
                noise_tr, std_dev_tr, z_tr = sample_noise(tr_g, dt_tr, tr_score.shape, device, add_noise=add_noise)
                noise_rot, std_dev_rot, z_rot = sample_noise(rot_g, dt_rot, rot_score.shape, device, add_noise=add_noise)
                noise_tor, std_dev_tor, z_tor = sample_noise(tor_g, dt_tor, tor_score.shape, device, add_noise=add_noise)

                # Compute actions
                action_tr = mean_tr + noise_tr
                action_rot = mean_rot + noise_rot
                action_tor = mean_tor + noise_tor



#                # trans
 #               if temp_sampling[0] != 1.0:
                if not no_temp:
                    tr_sigma_data = np.exp(0.48884149503636976 * np.log(model_args.tr_sigma_max) +
                                        (1 - 0.48884149503636976) * np.log(model_args.tr_sigma_min))
                    lambda_tr = (tr_sigma_data + sigma_tr) / (tr_sigma_data + sigma_tr / temp_sampling[0])
                    pre_scaled_mean_tr = (tr_g ** 2 * dt_tr * (lambda_tr + temp_sampling[0] * temp_psi[0] / 2))
                    mean_tr= (pre_scaled_mean_tr * tr_score)
                    std_dev_tr = (tr_g * np.sqrt(dt_tr * (1 + temp_psi[0])))
                    noise_tr = std_dev_tr*z_tr
                    action_tr = mean_tr + noise_tr

                # Rotation
#                if temp_sampling[1] != 1.0:

                    rot_sigma_data = np.exp(0.48884149503636976 * np.log(model_args.rot_sigma_max) +
                                            (1 - 0.48884149503636976) * np.log(model_args.rot_sigma_min))
                    lambda_rot = (rot_sigma_data + sigma_rot) / (rot_sigma_data + sigma_rot / temp_sampling[1])
                    pre_scaled_mean_rot = (rot_g ** 2 * dt_rot * (lambda_rot + temp_sampling[1] * temp_psi[1] / 2))
                    mean_rot= (pre_scaled_mean_rot * rot_score)
                    std_dev_rot = (rot_g * np.sqrt(dt_rot * (1 + temp_psi[1])))
                    noise_rot = std_dev_rot*z_rot
                    action_rot = mean_rot + noise_rot


                # Torsion
#                if temp_sampling[2] != 1.0:

                    tor_sigma_data = np.exp(0.48884149503636976 * np.log(model_args.tor_sigma_max) +
                                            (1 - 0.48884149503636976) * np.log(model_args.tor_sigma_min))
                    lambda_tor = (tor_sigma_data + sigma_tor) / (tor_sigma_data + sigma_tor / temp_sampling[2])
                    pre_scaled_mean_tor = (tor_g ** 2 * dt_tor * (lambda_tor + temp_sampling[2] * temp_psi[2] / 2))
                    mean_tor= (pre_scaled_mean_tor * tor_score)
                    std_dev_tor = (tor_g * np.sqrt(dt_tor * (1 + temp_psi[2])))
                    noise_tor = std_dev_tor*z_tor
                    action_tor = mean_tor + noise_tor

                action_tor_sliced = slice_torsion_updates(action_tor, complex_graph_batch)
                mean_tor_sliced = slice_torsion_updates(mean_tor, complex_graph_batch)
                noise_tor_sliced = slice_torsion_updates(noise_tor, complex_graph_batch)

                complex_graph_batch['ligand'].pos = modify_conformer_batch(
                    complex_graph_batch['ligand'].pos,
                    complex_graph_batch,
                    action_tr,
                    action_rot,
                    action_tor,
                    mask_rotate
                )

                # **Update visualization and build trajectory steps**
                for idx_b in range(b):
                    sample_idx = batch_id * batch_size + idx_b
                    if sample_idx >= N:
                        break
                    if visualization_list is not None:
                        visualization_list[sample_idx].add(
                            (complex_graph_batch['ligand'].pos[idx_b * n : n * (idx_b + 1)].detach().cpu()
                             + data_list[sample_idx].original_center.detach().cpu()),
                            part=1, order=t_idx + 2
                        )

                    start_idx = idx_b * n
                    end_idx   = (idx_b + 1) * n
                    ligand_pos_sample = complex_graph_batch['ligand'].pos[start_idx:end_idx].clone().cpu()





                    # Save only final `args.training_steps` steps
                    if t_idx >= inference_steps - args.training_steps:
                        # Prepare transformation data
                        transformation_data = {
                            'tr': {
                                'pre_scaled_mean': pre_scaled_mean_tr.item(),
                                'mean_old': mean_tr[idx_b].tolist(),
                                'std_dev': std_dev_tr.item(),
                                'action': action_tr[idx_b].tolist(),
                                'noise': noise_tr[idx_b].tolist(),   # store translation noise
                                'pre_scaled_mean_T': pre_scaled_mean_T_tr.item(),
                                'dt': dt_tr,
                                'current_t': tr_schedule[t_idx],
                                'sigma': sigma_tr,
                                'tr_g': tr_g,
                                'cum_scale': cum_scale_tr_step,
                            },
                            'rot': {
                                'pre_scaled_mean': pre_scaled_mean_rot.item(),
                                'mean_old': mean_rot[idx_b].tolist(),
                                'std_dev': std_dev_rot.item(),
                                'action': action_rot[idx_b].tolist(),
                                'noise': noise_rot[idx_b].tolist(),   # store rotation noise
                                'pre_scaled_mean_T': pre_scaled_mean_T_rot.item(),
                                'dt': dt_rot,
                                'current_t': rot_schedule[t_idx],
                                'sigma': sigma_rot,
                                'rot_g': rot_g,
                                'cum_scale': cum_scale_rot_step
                            },
                            'tor': {
                                'pre_scaled_mean': pre_scaled_mean_tor.item(),
                                'mean_old': mean_tor_sliced[idx_b].tolist(),
                                'std_dev': std_dev_tor.item(),
                                'action': action_tor_sliced[idx_b].tolist(),
                                'noise': noise_tor_sliced[idx_b].tolist(),  # store torsion noise (sliced)
                                'pre_scaled_mean_T': pre_scaled_mean_T_tor.item(),
                                'dt': dt_tor,
                                'current_t': tor_schedule[t_idx],
                                'sigma': sigma_tor,
                                'tor_g': tor_g,
                                'cum_scale': cum_scale_tor_step,


                            },
                            'complex_graph_for_model': save_mod_complex_graph_batch,
                            'complex_graph_for_slicing': complex_graph_batch
                        }

                        step = {
                            'transformation_data': transformation_data
                        }

                        # Add step/time info
                        step['transformation_data']['step_idx'] = t_idx
                        step['transformation_data']['inference_steps'] = inference_steps


                        # **Include ligand_pos only at final timestep**
                        if t_idx == inference_steps - 1:
                            original_center = data_list[sample_idx].original_center.detach().cpu()
                            ligand_pos_global = ligand_pos_sample + original_center
                            step['ligand_pos'] = ligand_pos_global.tolist()

                        #print(f"Appending step for sample {sample_idx} at time {t_idx}:")

                        # **Append the step to the trajectory**
                        trajectories[sample_idx].append(step)
                        # **Skip saving transformation data for earlier timesteps**
                        #print(f"Skipping transformation data for sample {sample_idx} at time {t_idx}")


    # After sampling, save visualization frames if required
    if visualization_list is not None:
        intermediates_dir = os.path.join(model_args.out_dir, 'intermediates')
        os.makedirs(intermediates_dir, exist_ok=True)  # Ensure the intermediates directory exists
        for idx, pdb in enumerate(visualization_list):
            pdb_filename = os.path.join(intermediates_dir, f'sample{idx + 1}_trajectory_no_temp.pdb')
            pdb.write(pdb_filename)


    avg_translation_score_x = float(np.mean(translation_score_x)) if translation_score_x else 0.0
    avg_translation_score_y = float(np.mean(translation_score_y)) if translation_score_y else 0.0
    avg_translation_score_z = float(np.mean(translation_score_z)) if translation_score_z else 0.0

    # Return a dictionary containing the three averages.
    avg_translation_scores = {
        "avg_translation_score_x": avg_translation_score_x,
        "avg_translation_score_y": avg_translation_score_y,
        "avg_translation_score_z": avg_translation_score_z
    }

    return trajectories, avg_translation_scores
