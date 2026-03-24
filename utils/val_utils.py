
import yaml
import argparse
import os
import torch
import copy
from functools import partial
import sys
from rdkit import Chem
from tqdm import tqdm
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
from utils.visualise import PDBFile
from datasets.process_mols import write_mol_with_coords
from src.compute_probability import compute_log_prob
from src.reward import compute_rewards
from src.sampling_val import sampling_val

REWARD_THRESHOLDS = [1.5, 1.46, 1.25, 1.1, 1.06, 1.0, 0.96, 0.95, 0.93, 0.92, 0.917, 0.916, 0.90, 0.89, 0.88, 0.86, 0.85, 0.81, 0.8, 0.77, 0.73, 0.75, 0.67, 0.66, 0.33, 0.34, 0]


class ValRewardTracker:
    def __init__(self):
        self.threshold_counts = {threshold: 0 for threshold in REWARD_THRESHOLDS}
        self.total_trajectories = 0

    def update(self, raw_reward: float):
        self.total_trajectories += 1
        raw_reward = round(raw_reward, 2)
        if raw_reward in self.threshold_counts:
            self.threshold_counts[raw_reward] += 1

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        for threshold in REWARD_THRESHOLDS:
            count = self.threshold_counts[threshold]
            metrics[f'count_raw_reward_{threshold:.2f}'] = count
        return metrics
    
class SkippedComplexTracker:
    def __init__(self):
        self.skipped_complexes = {}

    def add_skipped(self, pdbid: str, reason: str):
        self.skipped_complexes[pdbid] = reason

    def get_summary(self):
        return {"total_skipped": len(self.skipped_complexes),
                "skipped_details": self.skipped_complexes}


VAL_SAMPLES_PER_COMPLEX = 5


def trajectory_generation_val(loader,
                          model_traj_generation: torch.nn.Module,
                          args: argparse.Namespace,
                          device: torch.device,
                          t_to_sigma,
                          save_visulize: bool = False,
                          base_dir: Optional[str] = None,
                          num_complexes_to_sample: int = None,
                          mode: str = 'Val',
                          no_temp: bool = False
                         ) -> Tuple[List, Dict]:
    """
    Run inference for validation/test: generate VAL_SAMPLES_PER_COMPLEX trajectories per complex,
    compute PB rewards, and report metrics. Stops early for a complex if a PB-valid sample is found.
    No reward normalisation or trajectory collection.
    """
    if not hasattr(args, 'out_dir') or args.out_dir is None:
        args.out_dir = './output'
    os.makedirs(args.out_dir, exist_ok=True)

    num_complexes = args.num_complexes_to_sample if num_complexes_to_sample is None else num_complexes_to_sample
    print(f'num complexes is {num_complexes}')

    raw_rewards_all = []
    successful_complex_count = 0
    skipped_tracker = SkippedComplexTracker()
    val_total_samples = 0
    pb_valid_complexes = 0
    reward_tracker = ValRewardTracker()

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

        if mode == 'Test' and current_pdbid in SEEN_IDS:
            print(f"Skipping {current_pdbid} (already seen)")
            continue
        if current_pdbid in BLACKLIST:
            skipped_tracker.add_skipped(current_pdbid, "Blacklisted complex")
            continue
        if not orig_complex_graph.success[0]:
            skipped_tracker.add_skipped(current_pdbid, "Invalid complex")
            continue

        print(f"\n=== Processing {current_pdbid} ===")

        successful_samples = 0
        complex_pb_valid = False
        complex_failed = False

        while successful_samples < VAL_SAMPLES_PER_COMPLEX and not complex_pb_valid:
            for attempt in range(3):
                try:
                    sample_graph = copy.deepcopy(orig_complex_graph)
                    randomize_position([sample_graph],
                                       no_torsion=args.no_torsion,
                                       no_random=False,
                                       tr_sigma_max=args.tr_sigma_max,
                                       flexible_sidechains=False)

                    visualization_list = None
                    if save_visulize:
                        visualization_list = []
                        lig = orig_complex_graph.mol[0]
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add((orig_complex_graph['ligand'].pos[0] + orig_complex_graph.original_center[0]).detach().cpu(),
                                1, 0)
                        pdb.add((sample_graph['ligand'].pos + sample_graph.original_center).detach().cpu(),
                                part=1, order=1)
                        visualization_list.append(pdb)

                    tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')
                    rot_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')
                    tor_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

                    with torch.no_grad():
                        try:
                            trajs, _ = sampling_val(data_list=[sample_graph],
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
                                            temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor],
                                            visualization_list=visualization_list,
                                            args=args, mode=mode, no_temp=no_temp)
                        except Exception as e:
                            print("\n \n sampling issue:", e, '\n \n')
                            traceback.print_exc()
                            raise

                    if trajs is None or len(trajs) == 0:
                        raise ValueError("No trajectory generated???")

                    traj = trajs[0]
                    final_step = traj[-1]
                    ligand_pos = final_step.get('ligand_pos')
                    if ligand_pos is None:
                        raise ValueError("ligand_pos is None???")

                    lig = orig_complex_graph.mol[0]
                    mol_pred = Chem.RemoveAllHs(copy.deepcopy(lig))
                    mol_pred_pos = np.array(ligand_pos, dtype=np.double)
                    output_file = os.path.join(args.out_dir,
                                               f"{current_pdbid}_{successful_samples}_final_ligand.sdf")
                    write_mol_with_coords(mol_pred, mol_pred_pos, output_file)

                    pdbbind_directory = args.pdbbind_full_path
                    rec_path = os.path.join(pdbbind_directory, current_pdbid,
                                            f"{current_pdbid}_protein.pdb")
                    mol_true_path = os.path.join(pdbbind_directory, current_pdbid,
                                                 f"{current_pdbid}_ligand.sdf")

                    raw_reward = compute_rewards(
                        output_file, mol_true_path, rec_path, mode=mode
                    )

                    raw_rewards_all.append(raw_reward)
                    reward_tracker.update(raw_reward)
                    val_total_samples += 1

                    if math.isclose(raw_reward, 1.0, abs_tol=1e-6):
                        complex_pb_valid = True
                        pb_valid_complexes += 1
                        print('PB-valid sample found — stopping early for this complex')

                    print(f' {current_pdbid} Sample {successful_samples + 1}: raw reward = {raw_reward:.6f}')

                    if not save_visulize:
                        os.remove(output_file)

                    del traj
                    torch.cuda.empty_cache()
                    successful_samples += 1
                    break  # success — exit retry loop

                except Exception as e:
                    print(f"Sample attempt {attempt + 1}/3 failed for {current_pdbid}: {e}")
            else:
                # all 3 attempts failed for this sample — give up on this complex
                print(f"All 3 attempts failed for {current_pdbid}, skipping complex")
                skipped_tracker.add_skipped(current_pdbid, "max retries: all 3 attempts failed")
                complex_failed = True
                break

        if not complex_failed:
            successful_complex_count += 1
            print(f"Successfully processed {current_pdbid} ({successful_complex_count}/{num_complexes})")

    mean_reward = float(torch.tensor(raw_rewards_all).mean()) if raw_rewards_all else 0.0

    metrics = {
        "successful_complexes": successful_complex_count,
        "skipped_max_retries": sum("max retries" in reason for reason in skipped_tracker.skipped_complexes.values()),
        "skipped_total": len(skipped_tracker.skipped_complexes),
        "avg_raw_reward": mean_reward,
        "avg_norm_reward": 0.0,
        "pb_valid_complexes": pb_valid_complexes,
        "pb_valid_fraction": (pb_valid_complexes / successful_complex_count) if successful_complex_count else 0.0,
        "val_total_samples": val_total_samples,
    }
    metrics.update(reward_tracker.get_metrics())
    metrics["skipped_details"] = skipped_tracker.get_summary()["skipped_details"]
    print("Sampling skipped summary:", metrics["skipped_details"])

    return [], metrics
