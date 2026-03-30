import copy
import sys
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Any, Dict, List, Optional, Tuple

from RLDiff.utils.sampling_utils import (
    calculate_mean,
    sample_noise,
    slice_torsion_updates,
    handle_nans,
    compute_g,
    _precompute_cumulative_scales,
    modify_conformer_torsion_angles_batch,
    modify_conformer_batch
)

current_dir = os.path.dirname(os.path.abspath(__file__))
diffdock_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(diffdock_dir)
from utils.diffusion_utils import set_time

def build_step_dict_(
    *,
    # TR
    pre_scaled_mean_tr,
    mean_tr,
    std_dev_tr,
    action_tr,
    noise_tr,
    pre_scaled_mean_T_tr,
    dt_tr,
    current_t_tr,
    sigma_tr,
    tr_g,
    cum_scale_tr,
    # ROT
    pre_scaled_mean_rot,
    mean_rot,
    std_dev_rot,
    action_rot,
    noise_rot,
    pre_scaled_mean_T_rot,
    dt_rot,
    current_t_rot,
    sigma_rot,
    rot_g,
    cum_scale_rot,
    # TOR (already sliced!)
    pre_scaled_mean_tor,
    mean_tor_sliced,
    std_dev_tor,
    action_tor_sliced,
    noise_tor_sliced,
    pre_scaled_mean_T_tor,
    dt_tor,
    current_t_tor,
    sigma_tor,
    tor_g,
    cum_scale_tor,
    # graphs + indices
    complex_graph_for_model,    # CPU clone pre-action
    complex_graph_for_slicing,  # live graph object at save-time
    step_idx: int,
    inference_steps: int,
) -> Dict[str, Any]:
    """
    Canonical step-dict builder that matches your MAIN dict schema/casts exactly.
    """
    return {
        'transformation_data': {
            'tr': {
                'pre_scaled_mean': pre_scaled_mean_tr.item(),
                'mean_old': mean_tr.tolist(),
                'std_dev': float(std_dev_tr),
                'action': action_tr.tolist(),
                'noise':  noise_tr.tolist(),
                'pre_scaled_mean_T': pre_scaled_mean_T_tr.item(),
                'dt': dt_tr,
                'current_t': current_t_tr,
                'sigma': sigma_tr,
                'tr_g': tr_g,
                'cum_scale': cum_scale_tr,
            },
            'rot': {
                'pre_scaled_mean': pre_scaled_mean_rot.item(),
                'mean_old': mean_rot.tolist(),
                'std_dev': float(std_dev_rot),
                'action': action_rot.tolist(),
                'noise':  noise_rot.tolist(),
                'pre_scaled_mean_T': pre_scaled_mean_T_rot.item(),
                'dt': dt_rot,
                'current_t': current_t_rot,
                'sigma': sigma_rot,
                'rot_g': rot_g,
                'cum_scale': cum_scale_rot,
            },
            'tor': {
                'pre_scaled_mean': pre_scaled_mean_tor.item(),
                'mean_old': mean_tor_sliced.tolist(),
                'std_dev': float(std_dev_tor),
                'action': action_tor_sliced.tolist(),
                'noise':  noise_tor_sliced.tolist(),
                'pre_scaled_mean_T': pre_scaled_mean_T_tor.item(),
                'dt': dt_tor,
                'current_t': current_t_tor,
                'sigma': sigma_tor,
                'tor_g': tor_g,
                'cum_scale': cum_scale_tor,
            },
            'complex_graph_for_model': complex_graph_for_model,
            'complex_graph_for_slicing': complex_graph_for_slicing,
            'step_idx': step_idx,
            'inference_steps': inference_steps,
        }
    }



def _do_diffusion_step(
    *,
    graph,
    model,
    t_idx: int,
    tr_schedule,
    rot_schedule,
    tor_schedule,
    inference_steps: int,
    t_to_sigma,
    model_args,
    cum_scales,
    device,
    b: int,
    mode: str,
    no_temp: bool,
    temp_sampling,
    temp_psi,
    mask_rotate,
) -> Dict[str, Any]:
    """
    Single diffusion step:
      set_time -> forward -> compute means/noise/actions (temp scaling if enabled) ->
      slice torsions -> apply to graph

    Returns tensors needed to build downstream dicts via build_step_dict_().
    """
    # Schedule values + deltas
    t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
    if t_idx < inference_steps - 1:
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1]
    else:
        dt_tr = tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx]

    # Set timestep on graph
    set_time(
        graph,
        t=None,
        t_tr=t_tr,
        t_rot=t_rot,
        t_tor=t_tor,
        t_sidechain_tor=t_tor,
        batchsize=b,
        all_atoms=True,
        asyncronous_noise_schedule=False,
        device=device,
        include_miscellaneous_atoms=getattr(model_args, 'include_miscellaneous_atoms', False),
    )

    # Pre-action snapshot for dict parity
    pre_action_graph_cpu = graph.clone().cpu()

    # Forward
    tr_score, rot_score, tor_score, _ = model(graph)
    if tor_score.numel() == 0:
        raise ValueError("Torsion score is unexpectedly empty. Skipping this complex.")

    # NaN handling
    tr_score = handle_nans({'tr': tr_score})['tr']
    rot_score = handle_nans({'rot': rot_score})['rot']
    tor_score = handle_nans({'tor': tor_score})['tor']

    # Sigma + g
    sigma_tr, sigma_rot, sigma_tor, _ = t_to_sigma(t_tr, t_rot, t_tor, t_sc_tor=t_tor)
    tr_g = compute_g(
        sigma_tr,
        getattr(model_args, 'tr_sigma_max', 1.0),
        getattr(model_args, 'tr_sigma_min', 0.1)
    ).to(device)
    rot_g = compute_g(
        sigma_rot,
        getattr(model_args, 'rot_sigma_max', 1.0),
        getattr(model_args, 'rot_sigma_min', 0.1)
    ).to(device)
    tor_g = compute_g(
        sigma_tor,
        getattr(model_args, 'tor_sigma_max', 1.0),
        getattr(model_args, 'tor_sigma_min', 0.1)
    ).to(device)

    # Means (base)
    mean_tr_dict = calculate_mean(tr_score, tr_g, dt_tr, t_tr)
    mean_rot_dict = calculate_mean(rot_score, rot_g, dt_rot, t_rot)
    mean_tor_dict = calculate_mean(tor_score, tor_g, dt_tor, t_tor)

    mean_tr = mean_tr_dict['mean']
    pre_scaled_mean_tr = mean_tr_dict['pre_scaled_mean']
    pre_scaled_mean_T_tr = mean_tr_dict['pre_scaled_mean_T']

    mean_rot = mean_rot_dict['mean']
    pre_scaled_mean_rot = mean_rot_dict['pre_scaled_mean']
    pre_scaled_mean_T_rot = mean_rot_dict['pre_scaled_mean_T']

    mean_tor = mean_tor_dict['mean']
    pre_scaled_mean_tor = mean_tor_dict['pre_scaled_mean']
    pre_scaled_mean_T_tor = mean_tor_dict['pre_scaled_mean_T']

    cum_scale_tr, cum_scale_rot, cum_scale_tor = cum_scales

    # Noise
    is_last = (t_idx == inference_steps - 1)
    add_noise = not (mode == 'Val' and is_last)
    noise_tr, std_dev_tr, z_tr = sample_noise(tr_g, dt_tr, tr_score.shape, device, add_noise=add_noise)
    noise_rot, std_dev_rot, z_rot = sample_noise(rot_g, dt_rot, rot_score.shape, device, add_noise=add_noise)
    noise_tor, std_dev_tor, z_tor = sample_noise(tor_g, dt_tor, tor_score.shape, device, add_noise=add_noise)

    # Actions (base)
    action_tr = mean_tr + noise_tr
    action_rot = mean_rot + noise_rot
    action_tor = mean_tor + noise_tor

    # Temperature scaling (matches your MAIN block)
    if not no_temp:
        tr_sigma_data = np.exp(
            0.48884149503636976 * np.log(model_args.tr_sigma_max) +
            (1 - 0.48884149503636976) * np.log(model_args.tr_sigma_min)
        )
        lambda_tr = (tr_sigma_data + sigma_tr) / (tr_sigma_data + sigma_tr / temp_sampling[0])
        pre_scaled_mean_tr = (tr_g ** 2 * dt_tr * (lambda_tr + temp_sampling[0] * temp_psi[0] / 2))
        mean_tr = (pre_scaled_mean_tr * tr_score)
        std_dev_tr = (tr_g * np.sqrt(dt_tr * (1 + temp_psi[0])))
        noise_tr = std_dev_tr * z_tr
        action_tr = mean_tr + noise_tr

        rot_sigma_data = np.exp(
            0.48884149503636976 * np.log(model_args.rot_sigma_max) +
            (1 - 0.48884149503636976) * np.log(model_args.rot_sigma_min)
        )
        lambda_rot = (rot_sigma_data + sigma_rot) / (rot_sigma_data + sigma_rot / temp_sampling[1])
        pre_scaled_mean_rot = (rot_g ** 2 * dt_rot * (lambda_rot + temp_sampling[1] * temp_psi[1] / 2))
        mean_rot = (pre_scaled_mean_rot * rot_score)
        std_dev_rot = (rot_g * np.sqrt(dt_rot * (1 + temp_psi[1])))
        noise_rot = std_dev_rot * z_rot
        action_rot = mean_rot + noise_rot

        tor_sigma_data = np.exp(
            0.48884149503636976 * np.log(model_args.tor_sigma_max) +
            (1 - 0.48884149503636976) * np.log(model_args.tor_sigma_min)
        )
        lambda_tor = (tor_sigma_data + sigma_tor) / (tor_sigma_data + sigma_tor / temp_sampling[2])
        pre_scaled_mean_tor = (tor_g ** 2 * dt_tor * (lambda_tor + temp_sampling[2] * temp_psi[2] / 2))
        mean_tor = (pre_scaled_mean_tor * tor_score)
        std_dev_tor = (tor_g * np.sqrt(dt_tor * (1 + temp_psi[2])))
        noise_tor = std_dev_tor * z_tor
        action_tor = mean_tor + noise_tor

    # Slice torsions (save-time uses sliced in your MAIN dict)
    action_tor_sliced = slice_torsion_updates(action_tor, graph)
    mean_tor_sliced = slice_torsion_updates(mean_tor, graph)
    noise_tor_sliced = slice_torsion_updates(noise_tor, graph)

    # Apply to graph
    graph['ligand'].pos = modify_conformer_batch(
        graph['ligand'].pos,
        graph,
        action_tr,
        action_rot,
        action_tor,
        mask_rotate,
    )

    return {
        'pre_action_graph_cpu': pre_action_graph_cpu,
        'scores': {'tr': tr_score, 'rot': rot_score, 'tor': tor_score},

        # TR
        'pre_scaled_mean_tr': pre_scaled_mean_tr,
        'mean_tr': mean_tr,
        'std_dev_tr': std_dev_tr,
        'action_tr': action_tr,
        'noise_tr': noise_tr,
        'pre_scaled_mean_T_tr': pre_scaled_mean_T_tr,
        'dt_tr': dt_tr,
        'current_t_tr': tr_schedule[t_idx],
        'sigma_tr': sigma_tr,
        'tr_g': tr_g,
        'cum_scale_tr': cum_scale_tr[t_idx],

        # ROT
        'pre_scaled_mean_rot': pre_scaled_mean_rot,
        'mean_rot': mean_rot,
        'std_dev_rot': std_dev_rot,
        'action_rot': action_rot,
        'noise_rot': noise_rot,
        'pre_scaled_mean_T_rot': pre_scaled_mean_T_rot,
        'dt_rot': dt_rot,
        'current_t_rot': rot_schedule[t_idx],
        'sigma_rot': sigma_rot,
        'rot_g': rot_g,
        'cum_scale_rot': cum_scale_rot[t_idx],

        # TOR (sliced)
        'pre_scaled_mean_tor': pre_scaled_mean_tor,
        'mean_tor_sliced': mean_tor_sliced,
        'std_dev_tor': std_dev_tor,
        'action_tor_sliced': action_tor_sliced,
        'noise_tor_sliced': noise_tor_sliced,
        'pre_scaled_mean_T_tor': pre_scaled_mean_T_tor,
        'dt_tor': dt_tor,
        'current_t_tor': tor_schedule[t_idx],
        'sigma_tor': sigma_tor,
        'tor_g': tor_g,
        'cum_scale_tor': cum_scale_tor[t_idx],
    }

def _do_diffusion_step_branched(
    *,
    graph,
    model,
    t_idx: int,
    tr_schedule,
    rot_schedule,
    tor_schedule,
    inference_steps: int,
    t_to_sigma,
    model_args,
    cum_scales,
    device,
    b: int,
    mode: str,
    no_temp: bool,
    temp_sampling,
    temp_psi,
    mask_rotate,
    n_children: int,
):
    """
    Branched diffusion step (efficient):
      - Runs the model ONCE on the parent graph at this timestep.
      - Samples fresh noise n_children times and produces n_children *new* child graphs.
      - Does NOT mutate the input `graph`.

    Returns:
      List[Dict[str, Any]] of length n_children.
      Each element contains:
        - 'graph': the child graph after applying its sampled action
        - the same fields as _do_diffusion_step returns (so you can build_step_dict_ the same way)
          but with per-child action/noise (and per-child sliced torsions).
    """
    n_children = int(n_children)
    if n_children <= 0:
        return []
    #print(f'doing the branched one!, cool')
    # ---- schedule values + deltas ----
    t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
    if t_idx < inference_steps - 1:
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1]
    else:
        dt_tr = tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx]

    # ---- set time on PARENT (then we deepcopy it for children) ----
    set_time(
        graph,
        t=None,
        t_tr=t_tr,
        t_rot=t_rot,
        t_tor=t_tor,
        t_sidechain_tor=t_tor,
        batchsize=b,
        all_atoms=True,
        asyncronous_noise_schedule=False,
        device=device,
        include_miscellaneous_atoms=getattr(model_args, 'include_miscellaneous_atoms', False),
    )

    # Pre-action snapshot for dict parity (same for all children)
    pre_action_graph_cpu = graph.clone().cpu()

    # ---- forward ONCE ----
    tr_score, rot_score, tor_score, _ = model(graph)
    if tor_score.numel() == 0:
        raise ValueError("Torsion score is unexpectedly empty. Skipping this complex.")

    tr_score = handle_nans({'tr': tr_score})['tr']
    rot_score = handle_nans({'rot': rot_score})['rot']
    tor_score = handle_nans({'tor': tor_score})['tor']

    # ---- sigma + g ----
    sigma_tr, sigma_rot, sigma_tor, _ = t_to_sigma(t_tr, t_rot, t_tor, t_sc_tor=t_tor)

    tr_g = compute_g(
        sigma_tr,
        getattr(model_args, 'tr_sigma_max', 1.0),
        getattr(model_args, 'tr_sigma_min', 0.1)
    ).to(device)
    rot_g = compute_g(
        sigma_rot,
        getattr(model_args, 'rot_sigma_max', 1.0),
        getattr(model_args, 'rot_sigma_min', 0.1)
    ).to(device)
    tor_g = compute_g(
        sigma_tor,
        getattr(model_args, 'tor_sigma_max', 1.0),
        getattr(model_args, 'tor_sigma_min', 0.1)
    ).to(device)

    # ---- base means (deterministic given scores) ----
    mean_tr_dict  = calculate_mean(tr_score,  tr_g,  dt_tr,  t_tr)
    mean_rot_dict = calculate_mean(rot_score, rot_g, dt_rot, t_rot)
    mean_tor_dict = calculate_mean(tor_score, tor_g, dt_tor, t_tor)

    mean_tr = mean_tr_dict['mean']
    pre_scaled_mean_tr = mean_tr_dict['pre_scaled_mean']
    pre_scaled_mean_T_tr = mean_tr_dict['pre_scaled_mean_T']

    mean_rot = mean_rot_dict['mean']
    pre_scaled_mean_rot = mean_rot_dict['pre_scaled_mean']
    pre_scaled_mean_T_rot = mean_rot_dict['pre_scaled_mean_T']

    mean_tor = mean_tor_dict['mean']
    pre_scaled_mean_tor = mean_tor_dict['pre_scaled_mean']
    pre_scaled_mean_T_tor = mean_tor_dict['pre_scaled_mean_T']

    cum_scale_tr, cum_scale_rot, cum_scale_tor = cum_scales

    # ---- determine if we add noise at this timestep ----
    is_last = (t_idx == inference_steps - 1)
    add_noise = not (mode == 'Val' and is_last)

    # ---- if temp scaling is ON, overwrite the deterministic mean/std formulas once (still no z yet) ----
    # (z/noise/action will be per-child)
    if not no_temp:
        tr_sigma_data = np.exp(
            0.48884149503636976 * np.log(model_args.tr_sigma_max) +
            (1 - 0.48884149503636976) * np.log(model_args.tr_sigma_min)
        )
        lambda_tr = (tr_sigma_data + sigma_tr) / (tr_sigma_data + sigma_tr / temp_sampling[0])
        pre_scaled_mean_tr = (tr_g ** 2 * dt_tr * (lambda_tr + temp_sampling[0] * temp_psi[0] / 2))
        mean_tr = (pre_scaled_mean_tr * tr_score)
        std_dev_tr = (tr_g * np.sqrt(dt_tr * (1 + temp_psi[0])))

        rot_sigma_data = np.exp(
            0.48884149503636976 * np.log(model_args.rot_sigma_max) +
            (1 - 0.48884149503636976) * np.log(model_args.rot_sigma_min)
        )
        lambda_rot = (rot_sigma_data + sigma_rot) / (rot_sigma_data + sigma_rot / temp_sampling[1])
        pre_scaled_mean_rot = (rot_g ** 2 * dt_rot * (lambda_rot + temp_sampling[1] * temp_psi[1] / 2))
        mean_rot = (pre_scaled_mean_rot * rot_score)
        std_dev_rot = (rot_g * np.sqrt(dt_rot * (1 + temp_psi[1])))

        tor_sigma_data = np.exp(
            0.48884149503636976 * np.log(model_args.tor_sigma_max) +
            (1 - 0.48884149503636976) * np.log(model_args.tor_sigma_min)
        )
        lambda_tor = (tor_sigma_data + sigma_tor) / (tor_sigma_data + sigma_tor / temp_sampling[2])
        pre_scaled_mean_tor = (tor_g ** 2 * dt_tor * (lambda_tor + temp_sampling[2] * temp_psi[2] / 2))
        mean_tor = (pre_scaled_mean_tor * tor_score)
        std_dev_tor = (tor_g * np.sqrt(dt_tor * (1 + temp_psi[2])))

    # ---- otherwise (no_temp==True), std_dev depends on dt and g through sample_noise() ----
    # We still want std_dev_* for logging, and we still need z per child.
    # We'll call sample_noise once just to get std_dev + a z template; then resample z per child ourselves.
    if no_temp:
        _noise_tr, std_dev_tr, _z_tr = sample_noise(tr_g, dt_tr, tr_score.shape, device, add_noise=add_noise)
        _noise_rot, std_dev_rot, _z_rot = sample_noise(rot_g, dt_rot, rot_score.shape, device, add_noise=add_noise)
        _noise_tor, std_dev_tor, _z_tor = sample_noise(tor_g, dt_tor, tor_score.shape, device, add_noise=add_noise)

    # ---- create children (each child gets fresh z -> fresh noise -> fresh action) ----
    out = []
    for _ in range(n_children):
        child_graph = copy.deepcopy(graph)

        if add_noise:
            z_tr = torch.randn_like(tr_score, device=device)
            z_rot = torch.randn_like(rot_score, device=device)
            z_tor = torch.randn_like(tor_score, device=device)
        else:
            z_tr = torch.zeros_like(tr_score, device=device)
            z_rot = torch.zeros_like(rot_score, device=device)
            z_tor = torch.zeros_like(tor_score, device=device)

        noise_tr = std_dev_tr * z_tr
        noise_rot = std_dev_rot * z_rot
        noise_tor = std_dev_tor * z_tor

        action_tr = mean_tr + noise_tr
        action_rot = mean_rot + noise_rot
        action_tor = mean_tor + noise_tor

        # Slice torsions (logging uses sliced, apply uses full action_tor)
        action_tor_sliced = slice_torsion_updates(action_tor, child_graph)
        mean_tor_sliced   = slice_torsion_updates(mean_tor, child_graph)
        noise_tor_sliced  = slice_torsion_updates(noise_tor, child_graph)

        # Apply to CHILD graph
        child_graph['ligand'].pos = modify_conformer_batch(
            child_graph['ligand'].pos,
            child_graph,
            action_tr,
            action_rot,
            action_tor,
            mask_rotate,
        )

        out.append({
            'graph': child_graph,
            'pre_action_graph_cpu': pre_action_graph_cpu,
            'scores': {'tr': tr_score, 'rot': rot_score, 'tor': tor_score},

            # TR
            'pre_scaled_mean_tr': pre_scaled_mean_tr,
            'mean_tr': mean_tr,
            'std_dev_tr': std_dev_tr,
            'action_tr': action_tr,
            'noise_tr': noise_tr,
            'pre_scaled_mean_T_tr': pre_scaled_mean_T_tr,
            'dt_tr': dt_tr,
            'current_t_tr': tr_schedule[t_idx],
            'sigma_tr': sigma_tr,
            'tr_g': tr_g,
            'cum_scale_tr': cum_scale_tr[t_idx],

            # ROT
            'pre_scaled_mean_rot': pre_scaled_mean_rot,
            'mean_rot': mean_rot,
            'std_dev_rot': std_dev_rot,
            'action_rot': action_rot,
            'noise_rot': noise_rot,
            'pre_scaled_mean_T_rot': pre_scaled_mean_T_rot,
            'dt_rot': dt_rot,
            'current_t_rot': rot_schedule[t_idx],
            'sigma_rot': sigma_rot,
            'rot_g': rot_g,
            'cum_scale_rot': cum_scale_rot[t_idx],

            # TOR (sliced)
            'pre_scaled_mean_tor': pre_scaled_mean_tor,
            'mean_tor_sliced': mean_tor_sliced,
            'std_dev_tor': std_dev_tor,
            'action_tor_sliced': action_tor_sliced,
            'noise_tor_sliced': noise_tor_sliced,
            'pre_scaled_mean_T_tor': pre_scaled_mean_T_tor,
            'dt_tor': dt_tor,
            'current_t_tor': tor_schedule[t_idx],
            'sigma_tor': sigma_tor,
            'tor_g': tor_g,
            'cum_scale_tor': cum_scale_tor[t_idx],
        })

    return out



def sampling(
    data_list: List[Data],
    model: torch.nn.Module,
    inference_steps: int,
    tr_schedule: List[float],
    rot_schedule: List[float],
    tor_schedule: List[float],
    device: torch.device,
    t_to_sigma,
    model_args,
    batch_size: int = 32,
    temp_sampling: List[float] = [1, 1, 1],
    temp_psi: List[float] = [1, 1, 1],
    args=None,
    mode: str = 'Train',
    no_temp: bool = False,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, float]]:
    """
    Tree-only branching sampler.

    Branch at timesteps in {t_end+1, ..., t_start} where:
        t_start = args.branched_steps
        t_end   = args.branched_to

    At each branching timestep, each active node produces branches_per_t children.
    Every leaf is a full trajectory from 0..end.
    Branching divergence happens at the branching step via _do_diffusion_step_branched.
    """
    N = len(data_list)
    all_trajectories: List[List[Dict[str, Any]]] = []

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    mask_rotate = torch.from_numpy(data_list[0]['ligand'].mask_rotate[0]).to(device)


    cum_scales = _precompute_cumulative_scales(
        inference_steps, tr_schedule, rot_schedule, tor_schedule, t_to_sigma, model_args
    )

    branches_per_t = int(getattr(args, "branches_per_t", 1) or 1)
    branches_per_t = max(1, branches_per_t)

    # Tree branching uses branched_steps / branched_to measured FROM THE END
    K_from_end = int(getattr(args, "branched_steps", 0) or 0)
    K_to_end   = int(getattr(args, "branched_to", 0) or 0)

    K_from_end = max(0, min(K_from_end, inference_steps))
    K_to_end   = max(0, min(K_to_end, inference_steps))

    t_branch_start = inference_steps - K_from_end
    t_branch_stop  = inference_steps - K_to_end

    branch_window_tree = (
        set(range(t_branch_start, t_branch_stop))
        if t_branch_start < t_branch_stop else set()
    )



    step_args = dict(
        model=model,
        tr_schedule=tr_schedule,
        rot_schedule=rot_schedule,
        tor_schedule=tor_schedule,
        inference_steps=inference_steps,
        t_to_sigma=t_to_sigma,
        model_args=model_args,
        cum_scales=cum_scales,
        device=device,
        mode=mode,
        no_temp=no_temp,
        temp_sampling=temp_sampling,
        temp_psi=temp_psi,
        mask_rotate=mask_rotate,
    )

    with torch.no_grad():
        for batch_id, complex_graph_batch in enumerate(loader):
            b = complex_graph_batch.num_graphs
            n = len(complex_graph_batch['ligand'].pos) // b
            complex_graph_batch = complex_graph_batch.to(device)

            # MAIN trajectories (full length)
            main_trajs_by_sample = [[] for _ in range(b)]

            # Track per-sample MAIN path (tuple of child-choices at each branching timestep)
            # Convention: continuation child = 0; extra spawned children = 1..(B-1)
            main_paths_by_sample = [tuple() for _ in range(b)]

            # Active suffix nodes. Each node stores ONLY suffix steps since it was spawned.
            # node = {
            #   "graph": <device graph>,
            #   "steps_by_sample": [list per sample],
            #   "spawn_t": int,
            #   "path": tuple[int, ...],   # branch choices so far
            # }
            active_nodes: List[Dict[str, Any]] = []

            # How many *extra* children per parent? (Continuation already exists.)
            extra_children_per_parent = max(0, branches_per_t - 1)

            # We only record/extend paths when branching is actually meaningful
            has_branching = (branches_per_t > 1)

            for t_idx in range(inference_steps):

                is_branch_t = (t_idx in branch_window_tree) and has_branching

                # -------------------------
                # (1) STEP MAIN ONCE
                # -------------------------
                step_data_main = _do_diffusion_step(
                    graph=complex_graph_batch, t_idx=t_idx, b=b, **step_args
                )

                # Save MAIN step dicts
                for idx_b in range(b):
                    step = build_step_dict_(
                        pre_scaled_mean_tr=step_data_main['pre_scaled_mean_tr'],
                        mean_tr=step_data_main['mean_tr'][idx_b],
                        std_dev_tr=step_data_main['std_dev_tr'],
                        action_tr=step_data_main['action_tr'][idx_b],
                        noise_tr=step_data_main['noise_tr'][idx_b],
                        pre_scaled_mean_T_tr=step_data_main['pre_scaled_mean_T_tr'],
                        dt_tr=step_data_main['dt_tr'],
                        current_t_tr=step_data_main['current_t_tr'],
                        sigma_tr=step_data_main['sigma_tr'],
                        tr_g=step_data_main['tr_g'],
                        cum_scale_tr=step_data_main['cum_scale_tr'],

                        pre_scaled_mean_rot=step_data_main['pre_scaled_mean_rot'],
                        mean_rot=step_data_main['mean_rot'][idx_b],
                        std_dev_rot=step_data_main['std_dev_rot'],
                        action_rot=step_data_main['action_rot'][idx_b],
                        noise_rot=step_data_main['noise_rot'][idx_b],
                        pre_scaled_mean_T_rot=step_data_main['pre_scaled_mean_T_rot'],
                        dt_rot=step_data_main['dt_rot'],
                        current_t_rot=step_data_main['current_t_rot'],
                        sigma_rot=step_data_main['sigma_rot'],
                        rot_g=step_data_main['rot_g'],
                        cum_scale_rot=step_data_main['cum_scale_rot'],

                        pre_scaled_mean_tor=step_data_main['pre_scaled_mean_tor'],
                        mean_tor_sliced=step_data_main['mean_tor_sliced'][idx_b],
                        std_dev_tor=step_data_main['std_dev_tor'],
                        action_tor_sliced=step_data_main['action_tor_sliced'][idx_b],
                        noise_tor_sliced=step_data_main['noise_tor_sliced'][idx_b],
                        pre_scaled_mean_T_tor=step_data_main['pre_scaled_mean_T_tor'],
                        dt_tor=step_data_main['dt_tor'],
                        current_t_tor=step_data_main['current_t_tor'],
                        sigma_tor=step_data_main['sigma_tor'],
                        tor_g=step_data_main['tor_g'],
                        cum_scale_tor=step_data_main['cum_scale_tor'],

                        complex_graph_for_model=step_data_main['pre_action_graph_cpu'],
                        complex_graph_for_slicing=complex_graph_batch,
                        step_idx=t_idx,
                        inference_steps=inference_steps,
                    )

                    # ---- attach tree metadata for reward assignment ----
                    sample_root_id = batch_id * batch_size + idx_b
                    step["tree_root_id"] = int(sample_root_id)

                    # At a branch timestep, the "main" step corresponds to the continuation child (choice 0)
                    curr_path = main_paths_by_sample[idx_b] + ((0,) if is_branch_t else tuple())
                    step["tree_path"] = tuple(curr_path)

                    main_trajs_by_sample[idx_b].append(step)

                # If this is a branching timestep, update MAIN paths for future steps (continuation choice=0)
                if is_branch_t:
                    for idx_b in range(b):
                        main_paths_by_sample[idx_b] = main_paths_by_sample[idx_b] + (0,)

                # Attach final ligand_pos to MAIN
                if t_idx == inference_steps - 1:
                    for idx_b in range(b):
                        sample_idx = batch_id * batch_size + idx_b
                        ligand_pos_sample = complex_graph_batch['ligand'].pos[idx_b * n:(idx_b + 1) * n].clone().cpu()
                        if sample_idx < N:
                            ligand_pos_sample = ligand_pos_sample + data_list[sample_idx].original_center.detach().cpu()
                        main_trajs_by_sample[idx_b][-1]['ligand_pos'] = ligand_pos_sample.tolist()

                # -------------------------
                # (2) STEP ALL ACTIVE NODES ONCE
                # -------------------------
                # Keep the step_data for each node so we can branch from the *pre-step* state at this t_idx
                node_step_data: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []  # (node, step_data_node)

                for node in active_nodes:
                    step_data_node = _do_diffusion_step(
                        graph=node["graph"], t_idx=t_idx, b=b, **step_args
                    )
                    node_step_data.append((node, step_data_node))

                    for idx_b in range(b):
                        br_step = build_step_dict_(
                            pre_scaled_mean_tr=step_data_node['pre_scaled_mean_tr'],
                            mean_tr=step_data_node['mean_tr'][idx_b],
                            std_dev_tr=step_data_node['std_dev_tr'],
                            action_tr=step_data_node['action_tr'][idx_b],
                            noise_tr=step_data_node['noise_tr'][idx_b],
                            pre_scaled_mean_T_tr=step_data_node['pre_scaled_mean_T_tr'],
                            dt_tr=step_data_node['dt_tr'],
                            current_t_tr=step_data_node['current_t_tr'],
                            sigma_tr=step_data_node['sigma_tr'],
                            tr_g=step_data_node['tr_g'],
                            cum_scale_tr=step_data_node['cum_scale_tr'],

                            pre_scaled_mean_rot=step_data_node['pre_scaled_mean_rot'],
                            mean_rot=step_data_node['mean_rot'][idx_b],
                            std_dev_rot=step_data_node['std_dev_rot'],
                            action_rot=step_data_node['action_rot'][idx_b],
                            noise_rot=step_data_node['noise_rot'][idx_b],
                            pre_scaled_mean_T_rot=step_data_node['pre_scaled_mean_T_rot'],
                            dt_rot=step_data_node['dt_rot'],
                            current_t_rot=step_data_node['current_t_rot'],
                            sigma_rot=step_data_node['sigma_rot'],
                            rot_g=step_data_node['rot_g'],
                            cum_scale_rot=step_data_node['cum_scale_rot'],

                            pre_scaled_mean_tor=step_data_node['pre_scaled_mean_tor'],
                            mean_tor_sliced=step_data_node['mean_tor_sliced'][idx_b],
                            std_dev_tor=step_data_node['std_dev_tor'],
                            action_tor_sliced=step_data_node['action_tor_sliced'][idx_b],
                            noise_tor_sliced=step_data_node['noise_tor_sliced'][idx_b],
                            pre_scaled_mean_T_tor=step_data_node['pre_scaled_mean_T_tor'],
                            dt_tor=step_data_node['dt_tor'],
                            current_t_tor=step_data_node['current_t_tor'],
                            sigma_tor=step_data_node['sigma_tor'],
                            tor_g=step_data_node['tor_g'],
                            cum_scale_tor=step_data_node['cum_scale_tor'],

                            complex_graph_for_model=step_data_node['pre_action_graph_cpu'],
                            complex_graph_for_slicing=node["graph"],
                            step_idx=t_idx,
                            inference_steps=inference_steps,
                        )

                        # ---- attach tree metadata for reward assignment ----
                        sample_root_id = batch_id * batch_size + idx_b
                        br_step["tree_root_id"] = int(sample_root_id)

                        # At a branch timestep, this node's stepped update corresponds to continuation child (choice 0)
                        curr_path = node.get("path", tuple()) + ((0,) if is_branch_t else tuple())
                        br_step["tree_path"] = tuple(curr_path)

                        node["steps_by_sample"][idx_b].append(br_step)

                    # If this is a branching timestep, update node path for future steps (continuation choice=0)
                    if is_branch_t:
                        node["path"] = node.get("path", tuple()) + (0,)

                    if t_idx == inference_steps - 1:
                        for idx_b in range(b):
                            sample_idx = batch_id * batch_size + idx_b
                            ligand_pos_sample = node["graph"]['ligand'].pos[idx_b * n:(idx_b + 1) * n].clone().cpu()
                            if sample_idx < N:
                                ligand_pos_sample = ligand_pos_sample + data_list[sample_idx].original_center.detach().cpu()
                            node["steps_by_sample"][idx_b][-1]['ligand_pos'] = ligand_pos_sample.tolist()

                # -------------------------
                # (3) BRANCHING (COMPRESSED): spawn only (B-1) new suffix nodes per parent
                #     using the efficient branched step (1 forward per parent).
                # -------------------------
                if (t_idx in branch_window_tree) and (extra_children_per_parent > 0) and has_branching:

                    new_nodes: List[Dict[str, Any]] = []

                    # (3a) Branch from MAIN pre-step state (step_data_main['pre_action_graph_cpu'] is the pre-step snapshot)
                    main_parent = copy.deepcopy(step_data_main['pre_action_graph_cpu']).to(device)

                    main_children = _do_diffusion_step_branched(
                        graph=main_parent,
                        t_idx=t_idx,
                        b=b,
                        n_children=extra_children_per_parent,  # <-- KEY: only B-1
                        **step_args,
                    )

                    # Spawned MAIN children are choices 1..(B-1)
                    for child_choice, child_res in enumerate(main_children, start=1):
                        child_node = {
                            "graph": child_res["graph"],                 # already advanced at t_idx
                            "steps_by_sample": [[] for _ in range(b)],  # suffix-only log
                            "spawn_t": t_idx,
                            # NOTE: MAIN had its path updated with (0,) already at this timestep,
                            # but spawned children branch off the pre-step state at t_idx, so their path
                            # should be parent_path_before_this_step + (child_choice,)
                            "path": None,  # filled per-sample below when storing step dicts
                        }

                        # store the branch step (t_idx) immediately
                        for idx_b in range(b):
                            br_step = build_step_dict_(
                                pre_scaled_mean_tr=child_res['pre_scaled_mean_tr'],
                                mean_tr=child_res['mean_tr'][idx_b],
                                std_dev_tr=child_res['std_dev_tr'],
                                action_tr=child_res['action_tr'][idx_b],
                                noise_tr=child_res['noise_tr'][idx_b],
                                pre_scaled_mean_T_tr=child_res['pre_scaled_mean_T_tr'],
                                dt_tr=child_res['dt_tr'],
                                current_t_tr=child_res['current_t_tr'],
                                sigma_tr=child_res['sigma_tr'],
                                tr_g=child_res['tr_g'],
                                cum_scale_tr=child_res['cum_scale_tr'],

                                pre_scaled_mean_rot=child_res['pre_scaled_mean_rot'],
                                mean_rot=child_res['mean_rot'][idx_b],
                                std_dev_rot=child_res['std_dev_rot'],
                                action_rot=child_res['action_rot'][idx_b],
                                noise_rot=child_res['noise_rot'][idx_b],
                                pre_scaled_mean_T_rot=child_res['pre_scaled_mean_T_rot'],
                                dt_rot=child_res['dt_rot'],
                                current_t_rot=child_res['current_t_rot'],
                                sigma_rot=child_res['sigma_rot'],
                                rot_g=child_res['rot_g'],
                                cum_scale_rot=child_res['cum_scale_rot'],

                                pre_scaled_mean_tor=child_res['pre_scaled_mean_tor'],
                                mean_tor_sliced=child_res['mean_tor_sliced'][idx_b],
                                std_dev_tor=child_res['std_dev_tor'],
                                action_tor_sliced=child_res['action_tor_sliced'][idx_b],
                                noise_tor_sliced=child_res['noise_tor_sliced'][idx_b],
                                pre_scaled_mean_T_tor=child_res['pre_scaled_mean_T_tor'],
                                dt_tor=child_res['dt_tor'],
                                current_t_tor=child_res['current_t_tor'],
                                sigma_tor=child_res['sigma_tor'],
                                tor_g=child_res['tor_g'],
                                cum_scale_tor=child_res['cum_scale_tor'],

                                complex_graph_for_model=child_res['pre_action_graph_cpu'],
                                complex_graph_for_slicing=child_res["graph"],
                                step_idx=t_idx,
                                inference_steps=inference_steps,
                            )

                            # ---- attach tree metadata for reward assignment ----
                            sample_root_id = batch_id * batch_size + idx_b
                            br_step["tree_root_id"] = int(sample_root_id)

                            # parent_path BEFORE taking choice at this timestep = main_paths_by_sample[idx_b] WITHOUT the (0,) we appended
                            # We appended (0,) earlier at this same t_idx, so remove it for the branching parent reference.
                            parent_path_before = main_paths_by_sample[idx_b][:-1]  # safe because is_branch_t implies we appended
                            child_path = parent_path_before + (child_choice,)
                            br_step["tree_path"] = tuple(child_path)

                            child_node["steps_by_sample"][idx_b].append(br_step)


                        parent_path_before = main_paths_by_sample[0][:-1]  # remove the continuation (0,) appended at this t_idx
                        child_node["path"] = parent_path_before + (child_choice,)


                        new_nodes.append(child_node)

                    # (3b) Branch from EACH ACTIVE NODE pre-step state (captured in step_data_node['pre_action_graph_cpu'])
                    for (node, step_data_node) in node_step_data:
                        parent = copy.deepcopy(step_data_node['pre_action_graph_cpu']).to(device)

                        node_children = _do_diffusion_step_branched(
                            graph=parent,
                            t_idx=t_idx,
                            b=b,
                            n_children=extra_children_per_parent,  # <-- KEY: only B-1
                            **step_args,
                        )

                        # Spawned NODE children are choices 1..(B-1)
                        for child_choice, child_res in enumerate(node_children, start=1):
                            child_node = {
                                "graph": child_res["graph"],
                                "steps_by_sample": [[] for _ in range(b)],
                                "spawn_t": t_idx,
                                "path": None,  # set below
                            }

                            for idx_b in range(b):
                                br_step = build_step_dict_(
                                    pre_scaled_mean_tr=child_res['pre_scaled_mean_tr'],
                                    mean_tr=child_res['mean_tr'][idx_b],
                                    std_dev_tr=child_res['std_dev_tr'],
                                    action_tr=child_res['action_tr'][idx_b],
                                    noise_tr=child_res['noise_tr'][idx_b],
                                    pre_scaled_mean_T_tr=child_res['pre_scaled_mean_T_tr'],
                                    dt_tr=child_res['dt_tr'],
                                    current_t_tr=child_res['current_t_tr'],
                                    sigma_tr=child_res['sigma_tr'],
                                    tr_g=child_res['tr_g'],
                                    cum_scale_tr=child_res['cum_scale_tr'],

                                    pre_scaled_mean_rot=child_res['pre_scaled_mean_rot'],
                                    mean_rot=child_res['mean_rot'][idx_b],
                                    std_dev_rot=child_res['std_dev_rot'],
                                    action_rot=child_res['action_rot'][idx_b],
                                    noise_rot=child_res['noise_rot'][idx_b],
                                    pre_scaled_mean_T_rot=child_res['pre_scaled_mean_T_rot'],
                                    dt_rot=child_res['dt_rot'],
                                    current_t_rot=child_res['current_t_rot'],
                                    sigma_rot=child_res['sigma_rot'],
                                    rot_g=child_res['rot_g'],
                                    cum_scale_rot=child_res['cum_scale_rot'],

                                    pre_scaled_mean_tor=child_res['pre_scaled_mean_tor'],
                                    mean_tor_sliced=child_res['mean_tor_sliced'][idx_b],
                                    std_dev_tor=child_res['std_dev_tor'],
                                    action_tor_sliced=child_res['action_tor_sliced'][idx_b],
                                    noise_tor_sliced=child_res['noise_tor_sliced'][idx_b],
                                    pre_scaled_mean_T_tor=child_res['pre_scaled_mean_T_tor'],
                                    dt_tor=child_res['dt_tor'],
                                    current_t_tor=child_res['current_t_tor'],
                                    sigma_tor=child_res['sigma_tor'],
                                    tor_g=child_res['tor_g'],
                                    cum_scale_tor=child_res['cum_scale_tor'],

                                    complex_graph_for_model=child_res['pre_action_graph_cpu'],
                                    complex_graph_for_slicing=child_res["graph"],
                                    step_idx=t_idx,
                                    inference_steps=inference_steps,
                                )

                                # ---- attach tree metadata for reward assignment ----
                                sample_root_id = batch_id * batch_size + idx_b
                                br_step["tree_root_id"] = int(sample_root_id)

                                # node["path"] has already been updated with (0,) earlier at this t_idx (continuation),
                                # so parent path BEFORE choice = node["path"] without last element.
                                parent_path_before = tuple(node.get("path", tuple())[:-1])
                                child_path = parent_path_before + (child_choice,)
                                br_step["tree_path"] = tuple(child_path)

                                child_node["steps_by_sample"][idx_b].append(br_step)

                            child_node["path"] = parent_path_before + (child_choice,)
                            new_nodes.append(child_node)

                    # Add the new suffix nodes; they will be stepped on future timesteps (t_idx+1, ...)
                    active_nodes.extend(new_nodes)

                    # If branching at final step, attach ligand_pos immediately
                    # (these nodes won't be stepped again, so they need ligand_pos now)
                    if t_idx == inference_steps - 1:
                        for child_node in new_nodes:
                            for idx_b in range(b):
                                sample_idx = batch_id * batch_size + idx_b
                                ligand_pos_sample = child_node["graph"]['ligand'].pos[idx_b * n:(idx_b + 1) * n].clone().cpu()
                                if sample_idx < N:
                                    ligand_pos_sample = ligand_pos_sample + data_list[sample_idx].original_center.detach().cpu()
                                child_node["steps_by_sample"][idx_b][-1]['ligand_pos'] = ligand_pos_sample.tolist()

            # Collect: MAIN full + every node suffix (compressed)
            for idx_b in range(b):
                all_trajectories.append(main_trajs_by_sample[idx_b])
            for node in active_nodes:
                for idx_b in range(b):
                    all_trajectories.append(node["steps_by_sample"][idx_b])

    traj_lens = [len(tr) for tr in all_trajectories]
    if traj_lens:
        head = ", ".join(map(str, traj_lens[:8])) + (", ..." if len(traj_lens) > 8 else "")
        print(f"[sampling] strategy=tree returned {len(all_trajectories)} trajectories; lengths=[{head}]")        
        t0 = all_trajectories[0][0]['transformation_data']['step_idx']
        tN = all_trajectories[0][-1]['transformation_data']['step_idx']
        print(f"[sampling] traj[0]: steps={len(all_trajectories[0])} (t{t0}→t{tN}), final_has_ligand_pos={'ligand_pos' in all_trajectories[0][-1]}")
    avg_translation_scores = {
        "avg_translation_score_x": 0.0,
        "avg_translation_score_y": 0.0,
        "avg_translation_score_z": 0.0,
    }
    return all_trajectories, avg_translation_scores

