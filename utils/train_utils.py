import yaml
import argparse
import os
import torch
import copy
import sys
from rdkit import Chem
import numpy as np
from accelerate import Accelerator
from typing import List, Dict, Optional, Tuple
import gc
import random
import pandas as pd
from collections import defaultdict, deque
import math
import traceback


# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
diffdock_package_dir = os.path.join(parent_dir, 'DiffDock-Pocket')
if diffdock_package_dir not in sys.path:
    sys.path.append(diffdock_package_dir)

# Import necessary functions
from utils.sampling import randomize_position
from utils.diffusion_utils import get_t_schedule
from datasets.process_mols import write_mol_with_coords
from src.compute_probability import compute_log_prob
from src.reward import compute_rewards
from src.sampling import sampling

SEED = 42  # Set your desired seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

REWARD_THRESHOLDS = [1.5, 1.46, 1.25, 1.1, 1.06, 1.0, 0.96, 0.95, 0.93, 0.92, 0.917, 0.916, 0.90, 0.89, 0.88, 0.86, 0.85, 0.81, 0.8, 0.77, 0.73, 0.75, 0.67, 0.66, 0.33, 0.34, 0]
OPTIMAL_CHUNK_SIZES = {}
BLACKLIST = set()
# Global per-complex reward tracker
COMPLEX_REWARD_TRACKER = None


class RewardTracker:
    def __init__(self, initial_stats_file=None):
        self.threshold_counts = {threshold: 0 for threshold in REWARD_THRESHOLDS}
        self.total_trajectories = 0

        # Per-complex tracking
        self.complex_histories = defaultdict(lambda: deque(maxlen=4))
        self.initial_stats = {}

        # Load initial statistics if provided
        if initial_stats_file and os.path.exists(initial_stats_file):
            self.load_initial_stats(initial_stats_file)

    def load_initial_stats(self, csv_file):
        """Load initial statistics from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                complex_id = row['complex_id']
                mean_pb = row['mean_pb_score']
                self.initial_stats[complex_id] = {'mean': mean_pb}
            print(f"Loaded initial stats for {len(self.initial_stats)} complexes")
        except Exception as e:
            print(f"Warning: Could not load initial stats from {csv_file}: {e}")

    def get_complex_mean(self, complex_id):
        """Get current mean for a complex"""
        history = self.complex_histories[complex_id]

        if len(history) >= 5:
            mean_val = np.mean(list(history))
        elif len(history) > 0:
            mean_val =  np.mean(list(history))
        elif complex_id in self.initial_stats:
            mean_val =  self.initial_stats[complex_id]['mean']
        else:
            mean_val =  0.95
        return min(mean_val, 1.0)

    def update(self, complex_id, raw_reward: float):
        self.total_trajectories += 1
        raw_reward_rounded = round(raw_reward, 2)
        if raw_reward_rounded in self.threshold_counts:
            self.threshold_counts[raw_reward_rounded] += 1

        # Add to complex history
        self.complex_histories[complex_id].append(raw_reward)

    def get_metrics(self) -> Dict[str, float]:
        return {f'count_raw_reward_{k:.2f}': v for k, v in self.threshold_counts.items()}

class SkippedComplexTracker:
    def __init__(self):
        self.skipped_complexes = {}

    def add_skipped(self, pdbid: str, reason: str):
        self.skipped_complexes[pdbid] = reason

    def get_summary(self):
        return {"total_skipped": len(self.skipped_complexes),
                "skipped_details": self.skipped_complexes}


def _normalize_with_baseline(raw_list, eps: float = 1e-6):
    """
    Z-score normalize a list of raw x0 rewards for a single complex.
    Returns (normalized_list, mean, std).
    """
    arr = np.asarray(raw_list, dtype=np.float32)
    if arr.size == 0:
        return [], 0.0, 1.0
    mean = float(arr.mean())
    std  = float(arr.std())
    std  = max(std, eps) 
    z = (arr - mean) / std
    return z.tolist(), mean, std




def trajectory_generation(loader,
                          model_traj_generation: torch.nn.Module,
                          args: argparse.Namespace,
                          device: torch.device,
                          t_to_sigma,
                          max_retries: int = 10,
                          num_complexes_to_sample: int = None,
                          no_temp: bool = False
                         ) -> Tuple[List, Dict]:
    """
    Generate trajectories in phases:
    1. For each complex (loaded one at a time), generate trajectories one sample at a time.
       For each complex, attempt to generate args.samples_per_complex samples;
       if one sample fails, retry only that sample (up to max_retries per sample).
    2. Calculate raw rewards for final steps.
    3. Normalize rewards using statistics from all final steps.   (skipped in Val)
    4. Apply decay through trajectories.                          (skipped in Val)
    """
    # Ensure the output directory exists.
    if args.out_dir is None:
        args.out_dir = './output'
    os.makedirs(args.out_dir, exist_ok=True)


    num_complexes = args.num_complexes_to_sample if num_complexes_to_sample is None else num_complexes_to_sample

    print(f'num complexes is {num_complexes}')
    trajectories_all = []
    raw_rewards_all = []
    successful_complex_count = 0
    skipped_tracker = SkippedComplexTracker()
    global COMPLEX_REWARD_TRACKER
    if COMPLEX_REWARD_TRACKER is None:
        COMPLEX_REWARD_TRACKER = RewardTracker()

    reward_tracker = RewardTracker()
    reward_tracker.complex_histories = COMPLEX_REWARD_TRACKER.complex_histories
    reward_tracker.initial_stats = COMPLEX_REWARD_TRACKER.initial_stats


    data_iter = iter(loader)
    while successful_complex_count < num_complexes:
        try:
            orig_complex_graph = next(data_iter)
        except (StopIteration, RuntimeError, FileNotFoundError, EOFError, OSError) as e:
            print(f"[DataLoader] resetting iterator after {type(e).__name__}: {e}", flush=True)
            skipped_tracker.add_skipped("DATALOADER", f"{type(e).__name__}: {e}")
            data_iter = iter(loader)
            continue
        print(f"DEBUG ping: {orig_complex_graph.name[0]}", flush=True)

        current_pdbid = orig_complex_graph.name[0]

        if current_pdbid in BLACKLIST:
            skipped_tracker.add_skipped(current_pdbid, "Blacklisted complex")
            continue
        if not orig_complex_graph.success[0]:
            skipped_tracker.add_skipped(current_pdbid, "Invalid complex")
            continue

        print(f"\n=== Processing {current_pdbid} ===")

        # generating one by one
        complex_samples = []
        successful_samples = 0
        total_samples = args.samples_per_complex
        max_total_attempts = total_samples * max_retries
        total_attempts = 0
        #keep samplig until we get anough succesful samples to move into the training loop
        while successful_samples < total_samples and total_attempts < max_total_attempts:
            total_attempts += 1
            sample_success = False
            #we want to keep going until we have hit max_retries (so if we get errors in generation they don't count)
            for retry_count in range(max_retries):
                try:
                    sample_graph = copy.deepcopy(orig_complex_graph)
                    randomize_position([sample_graph],
                                       no_torsion=args.no_torsion,
                                       no_random=False,
                                       tr_sigma_max=args.tr_sigma_max,
                                       flexible_sidechains=False)


                    # Create schedules.
                    tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')
                    rot_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')
                    tor_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

                    with torch.no_grad():
                        try:
                            trajs, _ = sampling(data_list=[sample_graph],
                                            model=model_traj_generation,
                                            inference_steps=args.inference_steps,
                                            tr_schedule=tr_schedule,
                                            rot_schedule=rot_schedule,
                                            tor_schedule=tor_schedule,
                                            device=device,
                                            t_to_sigma=t_to_sigma,
                                            model_args=args,
                                            batch_size=args.batch_size,
                                            temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                                            temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                            #temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor],
                                            args=args, mode='Train', no_temp=no_temp)
                        except Exception as e:
                            print("\n \n sampling issue:", e, '\n \n')
                            traceback.print_exc()
                            raise

                    if not trajs:
                        raise ValueError("No trajectory generated???")

                    rollout_id = successful_samples  
                    for traj in trajs:
                        for step in traj:
                            step["tree_root_id"] = int(rollout_id)

                    leaf_rewards_this_sample = []
                    for leaf_idx, traj in enumerate(trajs):
                        final_step = traj[-1]
                        ligand_pos = final_step.get('ligand_pos')
                        if ligand_pos is None:
                            raise ValueError("ligand_pos is None???")

                        lig = orig_complex_graph.mol[0]
                        mol_pred = Chem.RemoveAllHs(copy.deepcopy(lig))
                        mol_pred_pos = np.array(ligand_pos, dtype=np.double)

                        # unique name per leaf
                        output_file = os.path.join(
                            args.out_dir,
                            f"{current_pdbid}_{successful_samples}_leaf{leaf_idx}_final_ligand.sdf"
                        )
                        write_mol_with_coords(mol_pred, mol_pred_pos, output_file)
                        pdbbind_directory = f"{parent_dir}/{args.pdbbind_dir}"
                        rec_path = os.path.join(pdbbind_directory, current_pdbid, f"{current_pdbid}_protein.pdb")
                        mol_true_path = os.path.join(pdbbind_directory, current_pdbid, f"{current_pdbid}_ligand.sdf")


                        raw_reward = compute_rewards(
                            output_file, mol_true_path, rec_path,
                            mode='Train'
                        )
                        os.remove(output_file)

                        raw_rewards_all.append(raw_reward)
                        reward_tracker.update(current_pdbid, raw_reward)

                        complex_samples.append((traj, raw_reward))

                        leaf_rewards_this_sample.append(raw_reward)


                    successful_samples += 1
                    print(f"Sample {successful_samples}/{total_samples} succeeded for {current_pdbid}")
                    sample_success = True
                    break
                except Exception as e:
                    print(f"Attempt {retry_count + 1} failed: {e}")
                    if retry_count == max_retries - 1:
                        BLACKLIST.add(current_pdbid)
                        skipped_tracker.add_skipped(current_pdbid, f"max retries: {str(e)}")

            if not sample_success:
                print(f"Failed to generate valid sample for {current_pdbid}, no reward")

        if successful_samples == total_samples:
            print(f"\n=== Processing {len(complex_samples)} leaves for {current_pdbid} ===")
            raw_list = [rw for (_traj, rw) in complex_samples]

            # Z-score across ALL x0 leaves for this complex
            norm_list, _, _ = _normalize_with_baseline(raw_list)

            # ---------------- reward assignment, tree-backup means by subtree ----------------
            z_all = norm_list

            T_inf = int(args.inference_steps)

            B = int(getattr(args, "branches_per_t", 1) or 1)
            B = max(1, B)

            K_from_end = int(getattr(args, "branched_steps", 0) or 0)
            K_to_end   = int(getattr(args, "branched_to", 0) or 0)

            K_from_end = max(0, min(K_from_end, T_inf))
            K_to_end   = max(0, min(K_to_end,   T_inf))

            t_branch_start = T_inf - K_from_end
            t_branch_stop  = T_inf - K_to_end  # exclusive

            branch_ts = list(range(t_branch_start, t_branch_stop)) if (t_branch_start < t_branch_stop) else []
            t_to_depth = {t: d for d, t in enumerate(branch_ts)}


            # ---- Collect per-leaf meta from the FINAL step (path at end of that returned trajectory) ----
            leaf_meta = []  # (root_id, path, z_leaf, traj, raw_reward)
            for (traj, raw_reward), z in zip(complex_samples, z_all):
                root_id = int(traj[-1]["tree_root_id"])
                path = traj[-1]["tree_path"]
                leaf_meta.append((root_id, path, float(z), traj, raw_reward))

            # ---- Build subtree mean tables ----
            from collections import defaultdict
            sums_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))

            for root_id, path, z_leaf, _traj, _raw in leaf_meta:
                # For every depth d that exists in this trajectory's path, add this leaf z to that prefix bucket.
                for d in range(len(path)):
                    pref = path[: d + 1]
                    sc = sums_counts[root_id][d][pref]
                    sc[0] += z_leaf
                    sc[1] += 1

            means = defaultdict(lambda: defaultdict(dict))
            for root_id in sums_counts:
                for d in sums_counts[root_id]:
                    for pref, (s, c) in sums_counts[root_id][d].items():
                        assert c > 0
                        means[root_id][d][pref] = s / c

            for root_id, path, _z_leaf, traj, _raw in leaf_meta:
                for step in traj:
                    si = step["transformation_data"]["step_idx"]
                    if si in t_to_depth:
                        d = t_to_depth[si]
                        assert len(path) >= (d + 1), (
                            f"[x0][tree] trajectory contains branch timestep t={si} (depth={d}) "
                            f"but final tree_path length={len(path)}. This is inconsistent."
                        )

            # ---- Assign rewards ----
            # Rule:
            #   - For t in branch_ts: reward signal = mean of z over descendants for the node at that depth (prefix mean)
            #   - For t < first branch timestep: use depth-0 subtree mean for this trajectory's first choice (expected future)
            #   - Otherwise: use leaf z (most specific)
            first_branch_t = branch_ts[0] if branch_ts else None

            for root_id, path, z_leaf, traj, raw_reward in leaf_meta:
                Tlen = len(traj)

                if first_branch_t is not None:
                    # To define "root expected value" we need at least one choice in the path.
                    # If path is empty, that means no branching choices were recorded; that's an error if branch_ts non-empty.
                    assert len(path) >= 1, (
                        f"[x0][tree] branch window is non-empty (first_branch_t={first_branch_t}) "
                        f"but trajectory has empty tree_path. Inconsistent sampler metadata."
                    )
                    root_val = means[root_id][0][path[:1]]
                else:
                    root_val = z_leaf  # no branch timesteps => degenerate

                for j, step in enumerate(traj):
                    si = step["transformation_data"]["step_idx"]
                    decay_pow = (Tlen - 1 - j)
                    decay = (args.decay_factor ** decay_pow)

                    if (first_branch_t is not None) and (si < first_branch_t):
                        used = root_val
                    elif si in t_to_depth:
                        d = t_to_depth[si]
                        pref = path[: d + 1]
                        used = means[root_id][d][pref]
                    else:
                        used = z_leaf

                    step["reward"] = used * decay

                    if j == Tlen - 1:
                        step["raw_reward"] = raw_reward

                trajectories_all.append(traj)

            successful_complex_count += 1
            print(f"Successfully processed {current_pdbid} ({successful_complex_count}/{num_complexes})")



    if raw_rewards_all:
        mean_reward = float(torch.tensor(raw_rewards_all).mean())
    else:
        mean_reward = 0.0

    total_norm_reward = 0
    if trajectories_all:
        for traj in trajectories_all:
            final_reward = traj[-1].get('reward', 0)
            decay = args.decay_factor ** (len(traj) - 1)
            original_norm_reward = final_reward / decay if decay != 0 else final_reward
            total_norm_reward += original_norm_reward

    metrics = {
        "successful_complexes": successful_complex_count,
        "skipped_max_retries": sum("max retries" in reason for reason in skipped_tracker.skipped_complexes.values()),
        "skipped_total": len(skipped_tracker.skipped_complexes),
        "avg_raw_reward": float(mean_reward),
        "avg_norm_reward": total_norm_reward / len(trajectories_all) if trajectories_all else 0.0
    }
    metrics.update(reward_tracker.get_metrics())
    metrics["skipped_details"] = skipped_tracker.get_summary()["skipped_details"]

    print("Sampling skipped summary:", metrics["skipped_details"])
    return trajectories_all, metrics




def train_with_samples(
    trajectories: List,
    model: torch.nn.Module,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device
) -> Tuple[float, float, dict]:
    """
    Perform offline training updates using pre-generated trajectories.
    Maintains original trajectories for all updates, only processing chunks as needed.
    Returns:
      avg_loss, avg_clipped_grad_norm, and a summary dictionary of complexes skipped during training.
    """
    total_loss = 0.0
    total_steps = 0
    total_clipped_grad_norm = 0.0
    grad_norm_steps = 0

    # Create a tracker for complexes skipped during training.
    training_skipped_tracker = SkippedComplexTracker()

    # Group trajectories by complex.
    complex_to_indices = {}
    for i, traj in enumerate(trajectories):
        complex_id = get_complex_id(traj)  # keep your existing helper
        if complex_id not in complex_to_indices:
            complex_to_indices[complex_id] = []
        complex_to_indices[complex_id].append(i)
    print(f"[train] complex_to_indices sizes: {{k: len(v) for k,v in complex_to_indices.items()}} ->",
          {k: len(v) for k, v in complex_to_indices.items()})

    done_with_update = False
    optimizer.zero_grad()

    # Check if all are blacklisted before we start this update.
    non_blacklisted = [
        cid for cid in complex_to_indices.keys()
        if cid not in BLACKLIST and complex_to_indices[cid]
    ]
    if not non_blacklisted:
        print("All complexes are blacklisted or empty?? Clearly an error somewhre")
        return 0.0, 0.0, training_skipped_tracker.get_summary()

    processed_complexes_this_update = set()

    for complex_id in non_blacklisted:
        if done_with_update:
            break
        if complex_id in processed_complexes_this_update:
            continue

        current_chunk_size = OPTIMAL_CHUNK_SIZES.get(complex_id, args.trajectory_chunk_size)
        print(f"[train] Starting with chunk_size={current_chunk_size} for complex {complex_id}")

        start_idx = 0
        indices = complex_to_indices[complex_id]
        while start_idx < len(indices):
            if done_with_update:
                break

            try:
                end_idx = min(start_idx + current_chunk_size, len(indices))
                chunk_indices = indices[start_idx:end_idx]
                trajectory_chunk = [trajectories[i] for i in chunk_indices]

                if not trajectory_chunk:
                    start_idx = end_idx
                    continue

                lens = [len(t) for t in trajectory_chunk]
                max_len_in_chunk = max(lens) if lens else 0
                print(f"[train]  chunk {start_idx}:{end_idx}  traj_lengths={lens}  max_len={max_len_in_chunk}")

                # === KEY CHANGE: window by actual max length in this chunk ===
                for step_start in range(0, max_len_in_chunk, args.step_chunk_size):
                    if done_with_update:
                        break
                    step_end = min(step_start + args.step_chunk_size, max_len_in_chunk)

                    # Only include non-empty slices for this window
                    step_chunks = [traj[step_start:step_end] for traj in trajectory_chunk if len(traj) > step_start]
                    print(f"[train]   window {step_start}:{step_end}  non_empty_slices={len(step_chunks)}")
                    if not step_chunks:
                        continue

                    try:
                        processed_steps = compute_log_prob(
                            trajectories=step_chunks,
                            model=model,
                            device=device,
                            args=args,
                            step_size=args.step_chunk_size,
                            step_start=step_start,
                            no_early_step_guidance=args.no_early_step_guidance,
                            alpha_step=args.alpha_step,
                        )
                        for processed_traj in processed_steps:
                            if done_with_update:
                                break

                            for processed_step in processed_traj:
                                try:
                                    with accelerator.accumulate(model):
                                        loss = compute_loss(processed_step, args.clip_range, args.no_early_step_guidance, args.alpha_step)
                                        total_loss += loss.item()
                                        total_steps += 1

                                        if total_steps % 10 == 1:
                                            print(f"[Step {total_steps}] loss={loss.item():.6f}")

                                        accelerator.backward(loss)

                                        if accelerator.sync_gradients:
                                            if args.max_grad_norm is not None:
                                                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                                                # track clipped norm
                                                gn2c = 0.0
                                                for p in model.parameters():
                                                    if p.grad is not None:
                                                        w = p.grad.data.norm(2).item()
                                                        if math.isfinite(w):
                                                            gn2c += w * w
                                                clipped_grad_norm = gn2c ** 0.5
                                                total_clipped_grad_norm += clipped_grad_norm
                                                grad_norm_steps += 1
                                                print(f"[Step {total_steps}] clip_grad_norm={clipped_grad_norm:.6f}")

                                            optimizer.step()
                                            optimizer.zero_grad()
                                            print(f"[Step {total_steps}] optimizer.step() + zero_grad()")

                                            print(f"\n=== Update step confirmed at step {total_steps} for complex {complex_id} ===\n")
                                            done_with_update = True
                                            break
                                finally:
                                    del processed_step
                            del processed_traj
                        del processed_steps

                    finally:
                        del step_chunks

                del trajectory_chunk
                start_idx = end_idx

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if 'trajectory_chunk' in locals():
                        del trajectory_chunk
                    if 'step_chunks' in locals():
                        del step_chunks
                    if 'processed_steps' in locals():
                        del processed_steps
                    cleanup()

                    if current_chunk_size != 1:
                        current_chunk_size = max(1, current_chunk_size // 2)
                        OPTIMAL_CHUNK_SIZES[complex_id] = current_chunk_size
                        print(f'[OOM] reduced chunk size -> {current_chunk_size}')
                    else:
                        current_chunk_size = 0
                        print(f'[OOM] blacklisting {complex_id}')
                        BLACKLIST.add(complex_id)
                        training_skipped_tracker.add_skipped(complex_id, "Chunk size is 0 due to CUDA OOM")
                        break

        if not done_with_update:
            processed_complexes_this_update.add(complex_id)
        cleanup()

    avg_loss = total_loss / max(total_steps, 1)
    avg_clipped_grad_norm = total_clipped_grad_norm / max(grad_norm_steps, 1)

    del trajectories
    cleanup()

    training_skipped_summary = training_skipped_tracker.get_summary()
    print("Training skipped summary:", training_skipped_summary)

    return avg_loss, avg_clipped_grad_norm, training_skipped_summary



def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_complex_id(trajectory):
    """Extract complex ID from trajectory data."""
    try:
        complex_graph = trajectory[0]['transformation_data']['complex_graph_for_model']
        if hasattr(complex_graph, 'name'):
            if isinstance(complex_graph.name, list) and complex_graph.name:
                name_value = complex_graph.name[0]
                return name_value[0] if isinstance(name_value, list) and name_value else str(name_value)
    except (KeyError, IndexError, AttributeError):
        pass
    return None


def compute_loss(step_data, clip_range, no_early_step_guidance, alpha_step):
    """Compute clipped loss for a single step, with a simple per-step early loss cap."""
    step = step_data['step']
    importance_weight = step_data['importance_weight']
    reward = step_data['reward']

    # print(f"[step {step}] [importance weight] is {importance_weight}")

    # Un/clipped PPO terms
    unclipped = importance_weight * reward
    # print(f"[unclipped] is {unclipped}")

    clipped_weights = torch.clamp(importance_weight, 1.0 - clip_range, 1.0 + clip_range)
    clipped = clipped_weights * reward

    if not torch.equal(unclipped, clipped):
        diff_mask = (unclipped != clipped)
        print(f"did a clip: from {unclipped[diff_mask][:5]} to {clipped[diff_mask][:5]}")


    # PPO-style loss
    loss = -torch.minimum(unclipped, clipped)

    # ---- per-step early loss cap (simple, hardcoded) ----
    if not no_early_step_guidance:
        if alpha_step is not None:
            if step < alpha_step:
                cap = torch.as_tensor(0.03, device=loss.device, dtype=loss.dtype)
                loss_abs = loss.abs()
                scale = torch.clamp(cap / (loss_abs + 1e-12), max=1.0)
                loss = loss * scale
        else:
            if step < 10:
                cap = torch.as_tensor(0.03, device=loss.device, dtype=loss.dtype)
                loss_abs = loss.abs()
                scale = torch.clamp(cap / (loss_abs + 1e-12), max=1.0)
                loss = loss * scale

    return loss


# Utility functions remain unchanged
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def dict_to_namespace(config_dict):
    args = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(args, key, value)
    return args

def save_model_parameters(model_dir, config):
    model_params = config.get('model', {})
    model_params_path = os.path.join(model_dir, 'model_parameters.yml')
    with open(model_params_path, 'w') as file:
        yaml.dump(model_params, file)
    print(f"Saved model_parameters.yml to {model_params_path}")

def log_metrics(wandb_run, epoch, metrics, *, commit=True):
    """
    Log `metrics` at a fixed step (`epoch`).  By default commit=True.
    """
    log_dict = {"epoch": epoch}
    log_dict.update(metrics)
    wandb_run.log(log_dict, step=epoch, commit=commit)


def save_checkpoint(model, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

def save_final_model(model, final_model_path):
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

def update_traj_model(model, model_traj_generation):
    """Safely update the trajectory generation model."""
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    model_traj_generation.load_state_dict(state_dict)
    model_traj_generation.eval()
    return model_traj_generation

def to_cpu(data):
    """Move data to CPU, handling nested structures including HeteroDataBatch."""
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(x) for x in data]
    elif hasattr(data, 'to'):  # HeteroDataBatch has a 'to' method
        return data.cpu()
    return data

def to_gpu(data, device):
    """Move data to GPU, handling nested structures including HeteroDataBatch."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_gpu(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(x, device) for x in data]
    elif hasattr(data, 'to'):  # HeteroDataBatch has a 'to' method
        return data.to(device)
    return data
