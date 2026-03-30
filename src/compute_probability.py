import torch
import math
from typing import List, Dict, Optional
from RLDiff.utils.compute_probability_utils import (
    compute_translation_likelihood,
    compute_rotation_likelihood,
    compute_torsion_likelihood,
    compute_optimal_score,
)
from RLDiff.utils.sampling_utils import (
    slice_torsion_updates,
    handle_nans,
)


def compute_step_probabilities(
    step: Dict,
    tr_score: torch.Tensor,
    rot_score: torch.Tensor,
    tor_score: torch.Tensor,
    device: torch.device,
    complex_graph,
    no_early_step_guidance: bool,
    args,
    alpha_step: int = None,
) -> Dict:
    """
    Compute probabilities for a single step.
    This version uses a blended action formulation:
        action = alpha * score_optimal + (1 - alpha) * (score + epsilon)
    where alpha decays from 1 (early) to 0 (late) according to a sigmoid schedule.

    Additionally, the reward is scaled as:
        scaled_reward = (1 - alpha) * reward + alpha * 1.0
    """
    from RLDiff.utils.compute_probability_utils import (
        compute_translation_likelihood,
        compute_rotation_likelihood,
        compute_torsion_likelihood,
    )

    step_data   = step['transformation_data']
    step_idx    = step_data['step_idx']
    total_steps = step_data['inference_steps']

    s      = 2
    t_mid  = total_steps / 2.0  # midpoint
    alpha  = 1.0 - (1.0 / (1.0 + math.exp(-s * (step_idx - t_mid))))



    if alpha_step is not None:
        alpha = 1.0 if step_idx < alpha_step else 0.0
    else:
        # default behavior (your previous 11-step cutoff)
        alpha = 1.0 if step_idx < 11 else 0.0



    print(f'step is {step_idx} and alpha is {alpha}')
    pre_scaled_mean_tr  = torch.as_tensor(step_data['tr']['pre_scaled_mean'],  device=device, dtype=torch.float32).unsqueeze(0)
    pre_scaled_mean_rot = torch.as_tensor(step_data['rot']['pre_scaled_mean'], device=device, dtype=torch.float32).unsqueeze(0)
    pre_scaled_mean_tor = torch.as_tensor(step_data['tor']['pre_scaled_mean'], device=device, dtype=torch.float32).unsqueeze(0)

    tr_cum  = torch.as_tensor(step_data['tr']['cum_scale'],  device=device, dtype=torch.float32).unsqueeze(0)
    rot_cum = torch.as_tensor(step_data['rot']['cum_scale'], device=device, dtype=torch.float32).unsqueeze(0)
    tor_cum = torch.as_tensor(step_data['tor']['cum_scale'], device=device, dtype=torch.float32).unsqueeze(0)

    mean_tr_new  = tr_score.unsqueeze(0)  * pre_scaled_mean_tr
    mean_rot_new = rot_score.unsqueeze(0) * pre_scaled_mean_rot
    mean_tor_new = tor_score.unsqueeze(0) * pre_scaled_mean_tor

    # OPTIMAL ACTION FOR EACH COMPONENT
    tr_transformation_optimal, rot_transformation_optimal, tor_transformation_optimal = compute_optimal_score(
        complex_graph, device, args,
        pre_scaled_mean_tr, pre_scaled_mean_tor, pre_scaled_mean_rot,
        tr_cum, tor_cum, rot_cum, mean_rot_new, mean_tor_new
    )

    # TRUE ACTION FOR EACH COMPONENT
    true_action_tr  = torch.as_tensor(step_data['tr']['action'],  device=device, dtype=torch.float32).unsqueeze(0)
    true_action_rot = torch.as_tensor(step_data['rot']['action'], device=device, dtype=torch.float32).unsqueeze(0)
    true_action_tor = torch.as_tensor(step_data['tor']['action'], device=device, dtype=torch.float32).unsqueeze(0)

    tr_optimal_magnitude = torch.norm(tr_transformation_optimal, p=2)
    tr_old_magnitude     = torch.norm(true_action_tr, p=2)

    # action = alpha * score_optimal + (1 - alpha) * (score + epsilon)
    if not no_early_step_guidance:
        tr_action  = alpha * tr_transformation_optimal  + (1 - alpha) * true_action_tr
        rot_action = alpha * rot_transformation_optimal + (1 - alpha) * true_action_rot
        tor_action = alpha * tor_transformation_optimal + (1 - alpha) * true_action_tor
    else:
        tr_action  = true_action_tr
        rot_action = true_action_rot
        tor_action = true_action_tor


    tr_std  = torch.as_tensor(step_data['tr']['std_dev'],  device=device, dtype=torch.float32).unsqueeze(0)
    rot_std = torch.as_tensor(step_data['rot']['std_dev'], device=device, dtype=torch.float32).unsqueeze(0)
    tor_std = torch.as_tensor(step_data['tor']['std_dev'], device=device, dtype=torch.float32).unsqueeze(0)


    log_p_theta_tr  = compute_translation_likelihood(tr_action,  mean_tr_new,  tr_std)
    log_p_theta_rot = compute_rotation_likelihood(  rot_action, mean_rot_new, rot_std)
    log_p_theta_tor = compute_torsion_likelihood(   tor_action, mean_tor_new, tor_std)
    print(f'LOG PROB FOR THE ROTATION NEW IS {  log_p_theta_rot}')
    print(f'LOG PROB FOR THE TRANS NEW IS {  log_p_theta_tr}')
    print(f'LOG PROB FOR THE TOR NEW IS {  log_p_theta_tor}')


    log_p_theta     = log_p_theta_tr + log_p_theta_rot + log_p_theta_tor
    log_p_theta     = torch.nan_to_num(log_p_theta, nan=-100.0, posinf=-100.0, neginf=-100.0)
    log_p_theta = torch.clamp(log_p_theta, -100, 100)

    with torch.no_grad():
        mean_tr_old  = torch.as_tensor(step_data['tr']['mean_old'],  device=device, dtype=torch.float32).unsqueeze(0)
        mean_rot_old = torch.as_tensor(step_data['rot']['mean_old'], device=device, dtype=torch.float32).unsqueeze(0)
        mean_tor_old = torch.as_tensor(step_data['tor']['mean_old'], device=device, dtype=torch.float32).unsqueeze(0)

        log_p_theta_old_tr  = compute_translation_likelihood(tr_action,  mean_tr_old,  tr_std)
        log_p_theta_old_rot = compute_rotation_likelihood(  rot_action, mean_rot_old, rot_std)
        log_p_theta_old_tor = compute_torsion_likelihood(   tor_action, mean_tor_old, tor_std)
        print(f'LOG PROB FOR THE ROTATION OLD IS {  log_p_theta_old_rot}')
        print(f'LOG PROB FOR THE TRANS OLD IS {  log_p_theta_old_tr}')
        print(f'LOG PROB FOR THE TOR OLD IS {  log_p_theta_old_tor}')

        log_p_theta_old     = log_p_theta_old_tr + log_p_theta_old_rot + log_p_theta_old_tor
        log_p_theta_old = torch.nan_to_num(log_p_theta_old, nan=-100.0, posinf=-100.0, neginf=-100.0)


    # Importance weight
    log_importance_weight = (log_p_theta - log_p_theta_old.detach())
    log_importance_weight = torch.clamp(log_importance_weight, -20.0, 20.0)

    print(f'log importance_weight is {log_importance_weight}')
    if torch.isnan(log_importance_weight).any() or torch.isinf(log_importance_weight).any():
        print("Warning: log_importance_weight contains nan/inf")
        print(f"log_p_theta stats: {log_p_theta.min():.4f}, {log_p_theta.max():.4f}")
        print(f"log_p_theta_old stats: {log_p_theta_old.min():.4f}, {log_p_theta_old.max():.4f}")

    importance_weight = log_importance_weight.exp()
    importance_weight = torch.nan_to_num(importance_weight, nan=0.0, posinf=torch.exp(torch.tensor(20.0, device=importance_weight.device)), neginf=0.0)

    print(f'importance weight is {importance_weight}')
    # ------------------------------------------------------------------
    # NEW: similarity-based early-step reward (range 0.25 → 4.0)
    # ------------------------------------------------------------------
    # Directional gap: 0 (same) … 1 (opposite)
    #dir_gap = (1.0 - torch.nn.functional.cosine_similarity(
    #    tr_transformation_optimal, mean_tr_old, dim=-1)) / 2.0

    # Magnitude gap: fractional absolute error, clipped to 1
    #mag_gap = torch.abs(torch.norm(tr_transformation_optimal, p=2) -
    #                    torch.norm(mean_tr_old,          p=2))
    #mag_gap = mag_gap / (torch.norm(tr_transformation_optimal, p=2) + 1e-6)
    #mag_gap = torch.clamp(mag_gap, max=1.0)

    # Combined distance 0…1
    #d_gap = 0.5 * (dir_gap + mag_gap)

    # Map to reward 0.25…4.0
    #gap_reward = 0 + 0.7 * d_gap
    # ------------------------------------------------------------------
    original_reward = step.get('reward')
    # Early (alpha≈1): largely gap_reward; Late (alpha≈0): env reward
    gap_reward = 0.03

    if not no_early_step_guidance:
        scaled_reward = (1 - alpha) * original_reward + alpha * gap_reward
    else:
        scaled_reward = original_reward


    return {
        'importance_weight': importance_weight,
        'log_p_theta':       log_p_theta,
        'reward':            scaled_reward,
        'optimal_magnitude': tr_optimal_magnitude,
        'old_magnitude':     tr_old_magnitude,
        'step': step_idx

    }



def process_trajectory_chunk(
    trajectory: List[Dict],
    start_idx: int,
    step_size: int,
    model: torch.nn.Module,
    device: torch.device,
    args,
    complex_graph_batch_model: Optional[torch.Tensor] = None,
    no_early_step_guidance: bool = False,
    alpha_step: int = None
) -> List[Dict]:
    """
    Process a chunk of steps from a single trajectory.
    """


    end_idx = min(start_idx + step_size, len(trajectory))
    step_chunk = trajectory[start_idx:end_idx]
    processed_steps = []


    # Get or use provided complex_graph_batch_model
    if complex_graph_batch_model is None:
        complex_graph_batch_model = step_chunk[0]['transformation_data']['complex_graph_for_model']
        complex_graph_batch_model = complex_graph_batch_model.to(device)

    # Get complex_graph_batch_slicing for torsion updates
    complex_graph_batch_slicing = step_chunk[0]['transformation_data']['complex_graph_for_slicing']
    complex_graph_batch_slicing = complex_graph_batch_slicing.to(device)


    # Forward pass through model
    tr_score_new, rot_score_new, tor_score_new = model(complex_graph_batch_model)[:3]
    # Handle NaNs in scores
    tr_score_new = handle_nans({'tr': tr_score_new})['tr']
    rot_score_new = handle_nans({'rot': rot_score_new})['rot']
    tor_score_new = handle_nans({'tor': tor_score_new})['tor']
    score_tor_sliced_new = slice_torsion_updates(tor_score_new, complex_graph_batch_slicing)

    for step_idx, step in enumerate(step_chunk):
        processed_step = compute_step_probabilities(
            step,
            tr_score_new[step_idx],
            rot_score_new[step_idx],
            score_tor_sliced_new[step_idx],
            device,
            complex_graph_batch_model,
            args=args,
            no_early_step_guidance=no_early_step_guidance,
            alpha_step=alpha_step
        )
        processed_steps.append(processed_step)

    # Clear memory
    del tr_score_new, rot_score_new, tor_score_new, score_tor_sliced_new
    torch.cuda.empty_cache()

    return processed_steps

def compute_log_prob(
    trajectories: List[List[Dict]],
    model: torch.nn.Module,
    device: torch.device,
    args,
    step_size: int = 2,
    step_start: int = None,
    no_early_step_guidance: bool = False,
    alpha_step: int = None
) -> List[List[Dict]]:
    """
    Compute log probabilities with memory-efficient chunking.
    """

    processed_trajectories = []

    # Process one trajectory at a time
    for traj_idx, trajectory in enumerate(trajectories):
        processed_steps = []

        # Process steps in chunks
        for step_start in range(0, len(trajectory), step_size):
            # Process chunk of steps
            chunk_steps = process_trajectory_chunk(
                trajectory=trajectory,
                start_idx=step_start,
                step_size=step_size,
                model=model,
                device=device,
                no_early_step_guidance=no_early_step_guidance,
                args=args,
                alpha_step=alpha_step,
            )
            processed_steps.extend(chunk_steps)

            # Clear memory between chunks
            torch.cuda.empty_cache()

        processed_trajectories.append(processed_steps)

        # Clear memory between trajectories
        if (traj_idx + 1) % 2 == 0:
            torch.cuda.empty_cache()

    return processed_trajectories
