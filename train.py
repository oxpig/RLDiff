# train.py
from pathlib import Path
import sys
from collections import OrderedDict
import os
import torch
from torch.optim import Adam
import yaml
import argparse
import copy
from functools import partial
import wandb
from argparse import Namespace
from accelerate import Accelerator
from accelerate.utils import set_seed
import gc
import random
import numpy as np
import logging

# Unused imports
#from tqdm import tqdm

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
diffdock_package_dir = os.path.join(parent_dir, 'DiffDock-Pocket')
if diffdock_package_dir not in sys.path:
    sys.path.append(diffdock_package_dir)

# DiffDock imports
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.download import download_and_extract

# RLDiff imports
from src.compute_probability import compute_log_prob
from src.reward import compute_rewards
from src.loader import construct_loaders_with_splits
from src.sampling import sampling

# Helper functions from train_utils
from RLDiff.utils.train_utils import (
    load_config,
    dict_to_namespace,
    save_model_parameters,
    log_metrics,
    save_checkpoint,
    save_final_model,
    update_traj_model,
    trajectory_generation,  # New: Sampling from a subset with success counting.
    train_with_samples,
    to_cpu,
    to_gpu,
)
from RLDiff.utils.val_utils import trajectory_generation_val

SEED = 42  # Choose your desired seed value

# 1. Python's built-in random module
random.seed(SEED)

# 2. NumPy
np.random.seed(SEED)

# 3. PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def branched_steps_counter(k: int) -> int:
    """
    Return the number of *extra* updates from branching the last k steps:
    k + (k-1) + ... + 1 = k*(k+1)//2
    """
    k = int(k or 0)
    if k < 0:
        raise ValueError(f"branched_steps must be >= 0, got {k}")
    return k * (k + 1) // 2


def train(config):
    """
    Orchestrates the RL training process using DDPO-IS with separate
    trajectory generation and offline training phases.
    Args:
        config (dict): Configuration parameters loaded from YAML.
    """
    base_dir = diffdock_package_dir
    args = dict_to_namespace(config)

    # Download models if necessary
    REPOSITORY_URL = 'https://github.com/plainerman/DiffDock-Pocket'
    REMOTE_URLS = [
        f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
        f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"
    ]

    os.makedirs(args.out_dir, exist_ok=True)

    args.model_dir = None
    args.model_cache_dir = '.cache/model'
    args.tag = 'v1.0.0'
    if args.model_dir is None:
        base_model_dir = os.path.join(args.model_cache_dir, args.tag)
        os.makedirs(base_model_dir, exist_ok=True)

        if args.model_dir is None:
            logging.debug(f'--model_dir is not set. Using tag: {args.tag}')
            args.model_dir = download_and_extract(f'{REPOSITORY_URL}/releases/download/{args.tag}/score_model.zip', base_model_dir, 'score_model')

        # if args.filtering_model_dir is None:
        #     logging.debug(f'--filtering_model_dir is not set. Using tag: {args.tag}')
        #     args.filtering_model_dir = download_and_extract(f'{REPOSITORY_URL}/releases/download/{args.tag}/confidence_model.zip', base_model_dir, 'confidence_model')



    # Set up device and memory optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Initialize data loaders
    dataset_root = f"{base_dir}/{args.pdbbind_dir}" if args.pdbbind_dir else None
    split_paths = {
        'train': args.split_train,
        'val':   args.split_val,
        'test':  args.split_test,
    }

    model_params_path = os.path.join(args.model_dir, 'model_parameters.yml')
    with open(model_params_path) as f:
        model_params = Namespace(**yaml.full_load(f))



    train_loader, val_loader, test_loader = construct_loaders_with_splits(
        dataset_root=dataset_root,
        split_paths=split_paths,
        args=args,
        score_model_args=model_params
    )
    print(f'\n \n \n THE TRAIN LOADER IS IS {train_loader} \n \n \n \n')
    print("→ train‐dataset size:", len(train_loader.dataset))
    print("→ train‐loader batches:", len(train_loader))

    branched_to_tag = "None" if getattr(args, "branched_to", None) is None else str(args.branched_to)

    if getattr(args, 'wandb', False):
        wandb_run = wandb.init(
            project=f"DD-Pocket-RL_paper_run, type={args.branching_strategy}",
            name=f"branched_from_{args.branched_steps}_until_{branched_to_tag}_with_{args.branches_per_t}_branches",
            config=config,
            mode="online"
        )
    else:
        wandb_run = None

    # Load model parameters and initialize models


    t_to_sigma = partial(t_to_sigma_compl, args=model_params)

    # ─── Curriculum scheduler ────────────────────────────────
    curriculum_level = 1
    target_mean_raw = 0.95      # promote when we reach this


    model_params.flexible_sidechains = True

    model = get_model(model_params, device, t_to_sigma=t_to_sigma, no_parallel=True)
    # NEW
    if args.state_dict is not None:
        print(f"→ Loading state_dict from {args.state_dict}")
        state_dict = torch.load(args.state_dict, map_location=device)
    else:
        print(f"→ Loading state_dict from default {args.model_dir}/{args.ckpt}")
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=device)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)


    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters:      {total_params:,}")

    # Create trajectory generation model BEFORE accelerator wrapping
    model_traj_generation = copy.deepcopy(model)
    model_traj_generation.eval()
    model.eval()

    branched_to_tag = "None" if getattr(args, "branched_to", None) is None else str(args.branched_to)
    checkpoint_dir = (
        f"{args.checkpoint_base_dir}_{args.branching_strategy}"
        f"_from_{args.branched_steps}_to_{branched_to_tag}"
        f"_with_{args.branches_per_t}_branches"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.w_decay
    )

    if args.optimizer is not None:
        print(f"→ Resuming optimizer state from {args.optimizer}")
        opt_state = torch.load(args.optimizer, map_location=device)
        optimizer.load_state_dict(opt_state)
    else:
        print("→ Using fresh optimizer (Adam)")


    branching_strategy = str(getattr(args, "branching_strategy", "normal") or "normal").lower()

    if branching_strategy == "tree":
        # Tree: branches_per_t^m leaves, where m = branched_steps - branched_to
        T = args.training_steps
        B = max(1, args.branches_per_t)
        K_from_end = max(0, min(int(args.branched_steps or 0), T))
        K_to_end   = max(0, min(int(args.branched_to or 0), T))
        t_branch_start = T - K_from_end
        t_branch_stop  = T - K_to_end
        m = max(0, t_branch_stop - t_branch_start)
        leaves_per_sample = B ** m if m > 0 else 1

        # Main trajectory: T steps
        # At each branching depth d, (B^d)*(B-1) new suffix nodes are spawned,
        # each running from t_branch_start+d to T-1, i.e. (T - t_branch_start - d) steps.
        total_suffix_steps = 0
        for d in range(m):
            num_new_nodes = (B ** d) * (B - 1)
            suffix_len = T - (t_branch_start + d)
            total_suffix_steps += num_new_nodes * suffix_len

        steps_per_sample = T + total_suffix_steps
        total_accumulation_steps = int(
            (args.num_complexes_to_sample * args.samples_per_complex * steps_per_sample)
            // args.num_updates_per_sampling
        )
    else:
        # Normal: triangular formula
        _extra_branch_updates = branched_steps_counter(args.branched_steps)
        steps_per_sample = int(args.training_steps + _extra_branch_updates)
        total_accumulation_steps = int(
            (args.num_complexes_to_sample * args.samples_per_complex * steps_per_sample)
            // args.num_updates_per_sampling
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=total_accumulation_steps,
        project_dir=checkpoint_dir,
        mixed_precision="no"
    )

    # Updated print statement that works for both strategies
    if branching_strategy == "tree":
        print(
            f"[accum] strategy=tree, T={T}, B={B}, branched_steps={args.branched_steps}, branched_to={args.branched_to} "
            f"=> t_branch=[{t_branch_start},{t_branch_stop}), m={m}, leaves_per_sample={leaves_per_sample}, "
            f"main_steps={T}, suffix_steps={total_suffix_steps}, steps_per_sample={steps_per_sample}; "
            f"total_accumulation_steps={total_accumulation_steps}"
        )
    else:
        print(
            f"[accum] strategy=normal, branched_steps={args.branched_steps} -> extra={_extra_branch_updates} "
            f"=> steps_per_sample={steps_per_sample} "
            f"(training_steps={args.training_steps} + triangular({args.branched_steps})={_extra_branch_updates}); "
            f"total_accumulation_steps={total_accumulation_steps}"
        )
    # Log accelerator configuration
    print(f"Gradient accumulation steps: {total_accumulation_steps}")
    # Prepare model and optimizer
    model, optimizer = accelerator.prepare(model, optimizer)

    history_rewards = []

    val_counter = 0

    # Training Loop
    for epoch in range(1, args.n_epochs + 1):
        print(f'\n=== Starting epoch {epoch} ===')

        # Phase 1: Generate trajectories
        model_traj_generation.eval()
        trajectories, traj_metrics = trajectory_generation(
            loader=train_loader,
            model_traj_generation=model_traj_generation,
            args=args,
            device=device,
            t_to_sigma=t_to_sigma,
            no_temp=args.no_temp
        )


        # Phase 2: Training
        model.eval()

        try:
            trajectories = to_cpu(trajectories)
            total_loss = 0
            training_skipped_list = []  # List to accumulate training-skipped summaries

            # Slice by sample groups (each sample produces identical trajectory structure)
            num_samples = args.num_complexes_to_sample * args.samples_per_complex
            trajs_per_sample = len(trajectories) // num_samples
            samples_per_slice = num_samples // args.num_updates_per_sampling
            assert num_samples % args.num_updates_per_sampling == 0, \
                f"num_samples={num_samples} not divisible by num_updates={args.num_updates_per_sampling}"

            slice_boundaries = [i * samples_per_slice * trajs_per_sample for i in range(args.num_updates_per_sampling)]
            slice_boundaries.append(len(trajectories))

            traj_step_counts = [len(t) for t in trajectories]

            # Process each slice
            for slice_idx in range(args.num_updates_per_sampling):
                start_idx = slice_boundaries[slice_idx]
                end_idx = slice_boundaries[slice_idx + 1]
                slice_steps = sum(traj_step_counts[start_idx:end_idx])
                print(f"  slice {slice_idx}: trajs [{start_idx}:{end_idx}], steps={slice_steps}, expected={total_accumulation_steps}")

                print(f"\n\n=== Processing slice {slice_idx + 1}/{args.num_updates_per_sampling}, including (indices {start_idx}:{end_idx}) ===\n\n")

                # Move slice to GPU and process
                gpu_slice = to_gpu(trajectories[start_idx:end_idx], device)
                slice_loss, epoch_grad_norm, slice_training_skipped = train_with_samples(
                    trajectories=gpu_slice,
                    model=model,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    args=args,
                    device=device,

                )
                total_loss += slice_loss
                training_skipped_list.append(slice_training_skipped)

                # Cleanup GPU memory
                del gpu_slice
                torch.cuda.empty_cache()
                gc.collect()
                # Log metrics and save
            epoch_loss = total_loss / args.num_updates_per_sampling
            metrics = {
                "train_loss": epoch_loss,
                "train_norm_reward": traj_metrics["avg_norm_reward"],
                "train_raw_reward": traj_metrics["avg_raw_reward"],
                "train_clipped_grad_norm": epoch_grad_norm,
                "training_skipped": training_skipped_list,  # aggregated training skipped details
                **traj_metrics,  # also includes sampling skipped details
                "curriculum_level": curriculum_level,
            }

            history_rewards.append(traj_metrics["avg_raw_reward"])
            window_size = 10
            history_rewards = history_rewards[-window_size:]

            # advance if the *average* of the window meets your bar
            if len(history_rewards) == window_size and np.mean(history_rewards) >= target_mean_raw:
                curriculum_level += 1
                print(f"🎉 Level up → {curriculum_level}!")



            if wandb_run:
                log_metrics(wandb_run, epoch, metrics, commit=False)


            if epoch % args.update_model_old_every == 0:
                model_traj_generation = update_traj_model(accelerator.unwrap_model(model), model_traj_generation)

            print(f"\nEpoch {epoch} - Loss: {epoch_loss:.4f}, Norm Reward: {traj_metrics['avg_norm_reward']:.4f}")


            if accelerator.is_main_process:
                # 1) save new ckpts (epoch-named)
                save_checkpoint(accelerator.unwrap_model(model), checkpoint_dir, epoch)

                opt_ckpt = os.path.join(checkpoint_dir, f"optimizer_epoch_{epoch}.pt")
                torch.save(optimizer.state_dict(), opt_ckpt)

                # 2) upload to W&B (keeps history there)
                if wandb_run:
                    wandb.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
                    wandb.save(opt_ckpt)

                # 3) local cleanup: keep ONLY the newest model + newest optimizer by mtime
                ckpt_path = Path(checkpoint_dir)

                # keep newest model file
                model_files = sorted(
                    ckpt_path.glob("model_epoch_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                for p in model_files[1:]:
                    p.unlink(missing_ok=True)

                # keep newest optimizer file
                opt_files = sorted(
                    ckpt_path.glob("optimizer_epoch_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                for p in opt_files[1:]:
                    p.unlink(missing_ok=True)

        finally:
            # Cleanup
            del trajectories
            torch.cuda.empty_cache()
            gc.collect()

        val_run = False

        if val_loader is not None and (epoch == 1 or epoch % args.do_val_run_every == 0):
            with torch.no_grad():
                model.eval()
                val_trajectories, val_traj_metrics = trajectory_generation_val(
                    loader=val_loader,
                    model_traj_generation=model_traj_generation,
                    args=args,
                    device=device,
                    t_to_sigma=t_to_sigma,
                    base_dir=base_dir,
                    num_complexes_to_sample=len(val_loader),
                    mode='Val',
                    no_temp=args.no_temp
                )
                print(f"\nEpoch {epoch} - Validation Trajectory Generation metrics: {val_traj_metrics}")

                if wandb_run:
                    val_log = {f"val_{k}": v for k, v in val_traj_metrics.items()}
                    log_metrics(wandb_run, epoch, val_log, commit=False)

                val_run = True
                val_counter += 1
                val_ckpt_name = f"val_model_{val_counter}.pt"
                val_ckpt_path = os.path.join(checkpoint_dir, val_ckpt_name)
                torch.save(accelerator.unwrap_model(model).state_dict(), val_ckpt_path)
                if wandb_run:
                    wandb.save(os.path.join(checkpoint_dir, val_ckpt_name))


                del val_trajectories
                # del val_traj_metrics

            # one unified log at the end of the epoch:

        final_log = {"epoch": epoch, **metrics}
        if val_run:
            final_log.update({f"val_{k}": v for k, v in val_traj_metrics.items()})
        if wandb_run:
            wandb_run.log(final_log, step=epoch)

        if val_run:
            del val_traj_metrics

        print(f"\nEpoch {epoch} completed. Training Loss: {epoch_loss:.4f}, Train Norm Reward: {traj_metrics['avg_norm_reward']:.4f}")


    # Testing Phase
    with torch.no_grad():
        model.eval()
        # For testing, we simply generate trajectories from the test loader
        test_trajectories, test_traj_metrics = trajectory_generation_val(
            loader=test_loader,
            model_traj_generation=model_traj_generation,
            args=args,
            device=device,
            t_to_sigma=t_to_sigma,
            no_temp=args.no_temp
        )
        print(f"Test Trajectory Generation metrics: {test_traj_metrics}")

        # Aggregate rewards (or other desired test metrics) from the test trajectories.
        test_total_norm_reward = 0.0
        test_total_raw_reward = 0.0
        test_count = 0
        for traj in test_trajectories:
            if traj and isinstance(traj, list) and len(traj) > 0:
                # Assuming the final step contains reward information
                test_total_norm_reward += traj[-1].get('reward', 0.0)
                test_total_raw_reward += traj[-1].get('raw_reward', 0.0)
                test_count += 1
        avg_test_norm_reward = test_total_norm_reward / test_count if test_count > 0 else 0.0
        avg_test_raw_reward = test_total_raw_reward / test_count if test_count > 0 else 0.0

    if accelerator.is_main_process:
        if wandb_run:
            log_metrics(wandb_run, args.n_epochs, {
                "test_norm_reward": avg_test_norm_reward,
                "test_raw_reward": avg_test_raw_reward,
                **test_traj_metrics
            })

        final_model_path = f"{checkpoint_dir}/final_model.pt"
        save_final_model(accelerator.unwrap_model(model), final_model_path)

    print("Training and Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model with DDPO-IS")
    parser.add_argument('--config', type=str, default='train_config.yaml',
                        help='Path to the training YAML configuration file')
    parser.add_argument('--state_dict', type=str, default=None,
                        help='Optional path to a model state_dict checkpoint (overrides model_dir+ckpt)')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='Optional path to an optimizer state_dict checkpoint (if resuming training)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--no_temp', action='store_true', default =False,
                        help='Disable temperature during trajectory generation.')
    parser.add_argument('--samples_per_complex', type=int, default=4,
                        help='samples_per_complex')
    parser.add_argument('--num_complexes_to_sample', type=int, default=12,
                        help='num_complexes_to_sample')
    parser.add_argument('--branched_steps', type=int, default=8,
                        help='branched_steps')
    parser.add_argument('--alpha_step', type=int, default=None,
                        help='alpha_step')
    parser.add_argument('--no_early_step_guidance', action='store_true',
        default=False, help='If set, do NOT use optimal/blended early-step actions; always use the true action.'
    )
    parser.add_argument('--branches_per_t', type=int, default=2,
                        help='branches_per_t (for branched sampling)')
    parser.add_argument('--branching_strategy', type=str, default='tree', choices=['normal', 'tree'],
                        help='Branching strategy: "normal" (last-K branches) or "tree" (branch off branches between branched_steps and branched_to).'
    )
    parser.add_argument(
        '--branched_to', type=int, default=4,
        help='(tree only) Stop branching at this step index (branching happens for steps t_end+1..t_start).'
    )



    cli_args = parser.parse_args()
    config_path = os.path.join(current_dir, cli_args.config)
    config = load_config(config_path)

    # Inject CLI args into config dict
    config["state_dict"] = cli_args.state_dict
    config["optimizer"] = cli_args.optimizer
    config["learning_rate"] = cli_args.learning_rate
    config["no_temp"] = cli_args.no_temp
    if cli_args.samples_per_complex is not None:
        config["samples_per_complex"] = cli_args.samples_per_complex
    if cli_args.num_complexes_to_sample is not None:
        config["num_complexes_to_sample"] = cli_args.num_complexes_to_sample
    if cli_args.branched_steps is not None:
        config["branched_steps"] = cli_args.branched_steps
    config["no_early_step_guidance"] = cli_args.no_early_step_guidance
    config["alpha_step"] = cli_args.alpha_step
    config["branches_per_t"] = cli_args.branches_per_t
    if cli_args.branching_strategy is not None:
        config["branching_strategy"] = cli_args.branching_strategy
    if cli_args.branched_to is not None:
        config["branched_to"] = cli_args.branched_to


    try:
        train(config)
    finally:
        wandb.finish()

