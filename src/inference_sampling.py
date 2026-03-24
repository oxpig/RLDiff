import copy
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
diffdock_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if diffdock_dir not in sys.path:
    sys.path.append(diffdock_dir)

from utils.diffusion_utils import modify_conformer, set_time


def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule,
             device, t_to_sigma, model_args,
             no_random=False, visualization_list=None,
             confidence_model=None, filtering_data_list=None, filtering_model_args=None,
             asyncronous_noise_schedule=False, t_schedule=None,
             batch_size=32, no_final_step_noise=False):
    """
    Simplified reverse diffusion sampling for RLDiff inference.
    Adapted from DiffDock-Pocket utils/sampling.py.
    No SVGD, no flexible sidechains, no temperature scaling.
    """
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        tr_score_list, rot_score_list, tor_score_list = [], [], []
        tr_sigma, rot_sigma, tor_sigma, _ = t_to_sigma(t_tr, t_rot, t_tor, t_tor)

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            set_time(complex_graph_batch,
                     t_schedule[t_idx] if t_schedule is not None else None,
                     t_tr, t_rot, t_tor, t_tor, b,
                     'all_atoms' in model_args and model_args.all_atoms,
                     asyncronous_noise_schedule, device,
                     include_miscellaneous_atoms=hasattr(model_args, 'include_miscellaneous_atoms') and model_args.include_miscellaneous_atoms)

            with torch.no_grad():
                tr_score, rot_score, tor_score, _ = model(complex_graph_batch)

            tr_score_list.append(tr_score.cpu())
            rot_score_list.append(rot_score.cpu())
            tor_score_list.append(tor_score.cpu())

        tr_score = torch.cat(tr_score_list, dim=0)
        rot_score = torch.cat(rot_score_list, dim=0)
        tor_score = torch.cat(tor_score_list, dim=0)

        tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
        rot_g = rot_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

        tr_z = torch.zeros((N, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
            else torch.normal(mean=0, std=1, size=(N, 3))
        tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

        rot_z = torch.zeros((N, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
            else torch.normal(mean=0, std=1, size=(N, 3))
        rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

        if not model_args.no_torsion:
            tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
            tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                else torch.normal(mean=0, std=1, size=tor_score.shape)
            tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
            torsions_per_molecule = tor_perturb.shape[0] // N
        else:
            tor_perturb = None

        data_list = [modify_conformer(complex_graph,
                                      tr_perturb[i:i + 1],
                                      rot_perturb[i:i + 1].squeeze(0),
                                      tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None)
                     for i, complex_graph in enumerate(data_list)]

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            filtering_loader = iter(DataLoader(filtering_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if filtering_data_list is not None:
                    filtering_complex_graph_batch = next(filtering_loader).to(device)
                    filtering_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(filtering_complex_graph_batch, 0, 0, 0, 0, 0, N, filtering_model_args.all_atoms,
                             asyncronous_noise_schedule, device,
                             include_miscellaneous_atoms=hasattr(filtering_model_args, 'include_miscellaneous_atoms') and filtering_model_args.include_miscellaneous_atoms)
                    confidence.append(confidence_model(filtering_complex_graph_batch))
                else:
                    set_time(complex_graph_batch, 0, 0, 0, 0, 0, N, filtering_model_args.all_atoms,
                             asyncronous_noise_schedule, device,
                             include_miscellaneous_atoms=hasattr(filtering_model_args, 'include_miscellaneous_atoms') and filtering_model_args.include_miscellaneous_atoms)
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence
