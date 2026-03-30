import copy
import functools
import logging
import os
import sys
import traceback
import tempfile
from argparse import ArgumentParser, Namespace, FileType
from functools import partial
from typing import Mapping, Optional

import torch
import yaml
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import RemoveHs
from tqdm import tqdm

# Directory setup - same as train.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
diffdock_package_dir = os.path.join(parent_dir, 'DiffDock-Pocket')
if diffdock_package_dir not in sys.path:
    sys.path.append(diffdock_package_dir)

from datasets.process_mols import write_mol_with_coords
from datasets.pdbbind import PDBBind, load_protein_ligand_df
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position
from utils.utils import get_model, get_available_devices, get_default_device, ensure_device
from utils.visualise import PDBFile
from utils.download import download_and_extract
from src.inference_sampling import sampling
from utils.minimize_utils import minimize_and_rerank

if os.name != 'nt':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

RDLogger.DisableLog('rdApp.*')

REPOSITORY_URL = 'https://github.com/plainerman/DiffDock-Pocket'
_BUILTIN_MODEL_PARAMS = os.path.join(current_dir, 'model_parameters.yml')


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--complex_name', type=str, default='unnamed_complex', help='Name that the complex will be saved with')
    parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
    parser.add_argument('--protein_path', '--experimental_protein', type=str, default=None, help='Path to the protein .pdb file')
    parser.add_argument('--ligand', type=str, default='COc(cc1)ccc1C#N', help='Either a SMILES string or the path to a molecule file that rdkit can read')
    parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
    parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
    parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

    parser.add_argument('--pocket_center_x', type=float, default=None, help='The x coordinate for the pocket center')
    parser.add_argument('--pocket_center_y', type=float, default=None, help='The y coordinate for the pocket center')
    parser.add_argument('--pocket_center_z', type=float, default=None, help='The z coordinate for the pocket center')

    parser.add_argument('--tag', type=str, default='v1.0.0', help='GitHub release tag that will be used to download a model if none is specified.')
    parser.add_argument('--model_cache_dir', type=str, default='.cache/model', help='Folder from where to load/restore the trained model')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
    parser.add_argument('--filtering_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--filtering_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--cache_path', type=str, default='.cache/data', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
    parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
    parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
    parser.add_argument('--inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')
    parser.add_argument('--inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')
    parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
    parser.add_argument('--keep_local_structures', action='store_true', default=True, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
    parser.add_argument('--skip_existing', action='store_true', default=False, help='If the output directory already exists, skip the inference')
    parser.add_argument('--state_dict', type=str, default=None, help='Path to trained RLDiff model state dict')
    parser.add_argument('--pocket_buffer', type=int, default=None, help='Override score_model_args.pocket_buffer at inference time.')

    parser.add_argument('--minimize_and_rerank', action='store_true', default=False,
                        help='After inference, merge outputs then minimize with smina (Vina) and rerank with GNINA')
    parser.add_argument('--smina_path', type=str, default='smina', help='Path or name of the smina executable')
    parser.add_argument('--gnina_path', type=str, default='gnina', help='Path or name of the gnina executable')
    parser.add_argument('--minimize_workers', type=int, default=4, help='Parallel workers for smina/gnina post-processing')

    return parser


@ensure_device
def infer_single_complex(idx: int, protein_ligand_info_row: Mapping, model: torch.nn.Module, args, score_model_args,
                         filtering_args=None, filtering_model=None, filtering_model_args=None,
                         filtering_complex_dict=None,
                         t_schedule=None, tr_schedule=None,
                         device=None):
    orig_complex_graph = protein_ligand_info_row["complex_graph"].to(device)

    complex_name = protein_ligand_info_row["complex_name"]
    spc = args.samples_per_complex

    rot_schedule = tr_schedule
    tor_schedule = tr_schedule

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    for m in (model, filtering_model):
        if m is not None:
            m = m.to(device)
            m.eval()

    if (filtering_model is not None and not (
            filtering_args.use_original_model_cache or filtering_args.transfer_weights) and complex_name
            not in filtering_complex_dict.keys()):
        print(f"HAPPENING | The filtering dataset did not contain {complex_name}. We are skipping this complex.")

    data_list = []
    try:
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(spc)]
        write_dir = f'{args.out_dir}/index{idx}___{complex_name.replace("/", "-")}'
        if os.path.exists(write_dir) and args.skip_existing:
            return 0

        randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max,
                           flexible_sidechains=False)

        lig = orig_complex_graph.mol
        if args.save_visualisation:
            visualization_list = []
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs:
                mol_pred = RemoveHs(mol_pred)
            for graph in data_list:
                pdb = PDBFile(mol_pred)
                pdb.add(mol_pred, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)
        else:
            visualization_list = None

        use_confidence = filtering_model is not None and not args.minimize_and_rerank
        if use_confidence and not (filtering_args.use_original_model_cache or filtering_args.transfer_weights):
            filtering_data_list = [copy.deepcopy(filtering_complex_dict[complex_name]) for _ in range(spc)]
        else:
            filtering_data_list = None

        data_list, confidence = sampling(data_list=data_list, model=model,
                                         inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                         tor_schedule=tor_schedule,
                                         t_schedule=t_schedule,
                                         t_to_sigma=t_to_sigma, model_args=score_model_args,
                                         confidence_model=filtering_model if use_confidence else None,
                                         device=device,
                                         visualization_list=visualization_list,
                                         no_random=args.no_random,
                                         filtering_data_list=filtering_data_list,
                                         filtering_model_args=filtering_model_args if use_confidence else None,
                                         asyncronous_noise_schedule=score_model_args.asyncronous_noise_schedule,
                                         batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise)

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
             for complex_graph in data_list])

        if confidence is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]
        os.makedirs(write_dir, exist_ok=True)
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs:
                mol_pred = RemoveHs(mol_pred)
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank + 1}.sdf'))
            if confidence is not None:
                write_mol_with_coords(mol_pred, pos,
                                      os.path.join(write_dir, f'rank{rank + 1}_confidence{confidence[rank]:.2f}.sdf'))

        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank + 1}_reverseprocess.pdb'))
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[rank].write(os.path.join(write_dir, f'rank{rank + 1}_reverseprocess.pdb'))

    except Exception as e:
        print("Failed on", complex_name, type(e))
        print(e)
        print(traceback.format_exc())
        return 0
    finally:
        del data_list

    return +1


@ensure_device
def infer_multiple_complexes(protein_ligand_df, *args, **kwargs):
    count_succeeded = 0
    num_input = protein_ligand_df.shape[0]
    with tqdm(total=num_input, desc="Docking inference") as pbar:
        for idx, protein_ligand_info_row in protein_ligand_df.iterrows():
            complex_name = protein_ligand_info_row["complex_name"]
            pbar.set_postfix_str(s=f"Row {idx}, complex {complex_name}", refresh=True)
            count_succeeded += infer_single_complex(idx, protein_ligand_info_row, *args, **kwargs)
            pbar.update()
    return count_succeeded


def main(args):
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    os.makedirs(args.out_dir, exist_ok=True)

    if args.filtering_model_dir is None:
        base_model_dir = os.path.join(args.model_cache_dir, args.tag)
        os.makedirs(base_model_dir, exist_ok=True)
        logging.debug(f'--filtering_model_dir is not set. Using tag: {args.tag}')
        args.filtering_model_dir = download_and_extract(f'{REPOSITORY_URL}/releases/download/{args.tag}/confidence_model.zip', base_model_dir, 'confidence_model')


    params_path = _BUILTIN_MODEL_PARAMS
    with open(params_path) as f:
        score_model_args = Namespace(**yaml.full_load(f))

    if args.pocket_buffer is not None:
        print(f"[OVERRIDE] pocket_buffer: {score_model_args.pocket_buffer} -> {args.pocket_buffer}", flush=True)
        score_model_args.pocket_buffer = args.pocket_buffer

    with open(f'{args.filtering_model_dir}/model_parameters.yml') as f:
        filtering_args = Namespace(**yaml.full_load(f))

    if args.protein_ligand_csv is not None:
        protein_ligand_df = load_protein_ligand_df(args.protein_ligand_csv, strict=False)
    elif args.protein_path is not None:
        if args.complex_name == 'unnamed_complex':
            args.complex_name = os.path.splitext(os.path.basename(args.protein_path))[0]
        df = pd.DataFrame({'complex_name': [args.complex_name],
                           'experimental_protein': [args.protein_path],
                           'ligand': [args.ligand],
                           'pocket_center_x': [args.pocket_center_x],
                           'pocket_center_y': [args.pocket_center_y],
                           'pocket_center_z': [args.pocket_center_z],
                           'flexible_sidechains': [False]})
        protein_ligand_df = load_protein_ligand_df(None, df=df)
    else:
        raise ValueError('Either --protein_ligand_csv or --protein_path has to be specified')

    if "computational_protein" in protein_ligand_df.columns:
        print("WARN: Dropping the column 'computational_protein' from the dataframe. "
              "This column is only used during training and will be ignored during inference.")
        protein_ligand_df.drop(columns=["computational_protein"], inplace=True)

    device = get_default_device()
    print(f"DiffDock-Pocket default device: {device}")

    os.makedirs(args.cache_path, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.cache_path) as dataset_cache:
        dataset_cache = os.path.join(dataset_cache, 'testset')
        print(f"[DATASET] all_atoms={score_model_args.all_atoms} | include_miscellaneous_atoms={score_model_args.include_miscellaneous_atoms}")
        test_dataset = PDBBind(transform=None,
                               protein_ligand_df=protein_ligand_df,
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
                               num_workers=args.num_workers,
                               keep_local_structures=args.keep_local_structures,
                               pocket_reduction=score_model_args.pocket_reduction,
                               pocket_buffer=score_model_args.pocket_buffer,
                               pocket_cutoff=score_model_args.pocket_cutoff,
                               pocket_reduction_mode=score_model_args.pocket_reduction_mode,
                               flexible_sidechains=False,
                               flexdist=score_model_args.flexdist,
                               flexdist_distance_metric=score_model_args.flexdist_distance_metric,
                               fixed_knn_radius_graph=not score_model_args.not_fixed_knn_radius_graph,
                               knn_only_graph=not score_model_args.not_knn_only_graph,
                               include_miscellaneous_atoms=score_model_args.include_miscellaneous_atoms,
                               use_old_wrong_embedding_order=score_model_args.use_old_wrong_embedding_order)

        filtering_test_dataset = filtering_complex_dict = None
        if args.filtering_model_dir is not None and not args.minimize_and_rerank:
            if not (filtering_args.use_original_model_cache or filtering_args.transfer_weights):
                print('HAPPENING | filtering model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the filtering model now.')
                filtering_test_dataset = PDBBind(transform=None,
                                                 protein_ligand_df=protein_ligand_df,
                                                 chain_cutoff=np.inf,
                                                 receptor_radius=filtering_args.receptor_radius,
                                                 cache_path=dataset_cache,
                                                 remove_hs=filtering_args.remove_hs,
                                                 max_lig_size=None,
                                                 c_alpha_max_neighbors=filtering_args.c_alpha_max_neighbors,
                                                 matching=False,
                                                 keep_original=False,
                                                 conformer_match_sidechains=False,
                                                 use_original_conformer_fallback=True,
                                                 popsize=filtering_args.matching_popsize,
                                                 maxiter=filtering_args.matching_maxiter,
                                                 all_atoms=filtering_args.all_atoms,
                                                 require_ligand=True,
                                                 num_workers=args.num_workers,
                                                 keep_local_structures=args.keep_local_structures,
                                                 pocket_reduction=filtering_args.pocket_reduction,
                                                 pocket_buffer=filtering_args.pocket_buffer,
                                                 pocket_cutoff=filtering_args.pocket_cutoff,
                                                 pocket_reduction_mode=filtering_args.pocket_reduction_mode,
                                                 flexible_sidechains=False,
                                                 flexdist=filtering_args.flexdist,
                                                 flexdist_distance_metric=filtering_args.flexdist_distance_metric,
                                                 fixed_knn_radius_graph=not filtering_args.not_fixed_knn_radius_graph,
                                                 knn_only_graph=not filtering_args.not_knn_only_graph,
                                                 include_miscellaneous_atoms=filtering_args.include_miscellaneous_atoms,
                                                 use_old_wrong_embedding_order=filtering_args.use_old_wrong_embedding_order)
                filtering_complex_dict = {d.name: d for d in filtering_test_dataset}

    if args.state_dict is not None:
            state_dict = args.state_dict
    else:
        RLDIFF_MODEL_URL = 'https://github.com/oxpig/RLDiff/releases/download/v1.0.0/DD_Pocket_RL_score_model.pt'
        rldiff_cache_dir = os.path.join(args.model_cache_dir, 'DD_Pocket_RL_score_model')
        os.makedirs(rldiff_cache_dir, exist_ok=True)
        state_dict = os.path.join(rldiff_cache_dir, 'DD_Pocket_RL_score_model.pt')
        if not os.path.exists(state_dict):
            print(f"Downloading DiffDock Pocket RL score model from {RLDIFF_MODEL_URL} ...")
            import urllib.request
            urllib.request.urlretrieve(RLDIFF_MODEL_URL, state_dict)
            print(f"Saved to {state_dict}")
        else:
            print(f"DiffDock Pocket RL score model already cached at {state_dict}")

    model_args = copy.deepcopy(score_model_args)
    model_args.flexible_sidechains = True

    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    model = get_model(model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    model.load_state_dict(torch.load(state_dict, map_location=device), strict=True)
    model = model.to(device)
    model.eval()

    if args.filtering_model_dir is not None and not args.minimize_and_rerank:
        if filtering_args.transfer_weights:
            with open(f'{filtering_args.original_model_dir}/model_parameters.yml') as f:
                filtering_model_args = Namespace(**yaml.full_load(f))
        else:
            filtering_model_args = filtering_args

        filtering_model = get_model(filtering_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
        state_dict = torch.load(f'{args.filtering_model_dir}/{args.filtering_ckpt}', map_location=torch.device('cpu'))
        filtering_model.load_state_dict(state_dict, strict=True)
        filtering_model = filtering_model.to(device)
        filtering_model.eval()
    else:
        filtering_model = None
        filtering_args = None
        filtering_model_args = None

    t_max = 1
    tr_schedule = get_t_schedule(sigma_schedule=args.sigma_schedule, inference_steps=args.inference_steps,
                                 inf_sched_alpha=args.inf_sched_alpha, inf_sched_beta=args.inf_sched_beta,
                                 t_max=t_max)
    t_schedule = None
    print('common tr schedule', tr_schedule)

    print('Size of test dataset: ', len(test_dataset))

    devices = get_available_devices(max_devices=args.num_workers)
    num_processes = len(devices)
    chunks = np.array_split(test_dataset.protein_ligand_df, num_processes)

    process_chunk = functools.partial(infer_multiple_complexes, model=model, args=args,
                                      score_model_args=score_model_args,
                                      filtering_args=filtering_args, filtering_model=filtering_model,
                                      filtering_model_args=filtering_model_args,
                                      filtering_complex_dict=filtering_complex_dict,
                                      t_schedule=t_schedule, tr_schedule=tr_schedule)

    if num_processes > 1:
        print(f"Starting {num_processes} processes.")
        with torch.multiprocessing.Pool(processes=num_processes) as pool:
            a_results = []
            for device, chunk in zip(devices, chunks):
                print(f"Starting process on device {device}")
                async_result = pool.apply_async(process_chunk, (chunk,), {"device": device})
                a_results.append(async_result)

            del test_dataset.protein_ligand_df, test_dataset, chunks, chunk
            pool.close()
            pool.join()

        print(f"Completed inferences")
    else:
        num_inferences = process_chunk(test_dataset.protein_ligand_df, device=device)
        print(f"Completed {num_inferences} / {len(test_dataset)} inferences")

    print(f'Results are in {args.out_dir}')

    if args.minimize_and_rerank:
        # Free GPU memory before launching minimize workers
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()

        protein_path_map = {
            str(row['complex_name']).replace('/', '-'): str(row['experimental_protein'])
            for _, row in protein_ligand_df.iterrows()
            if pd.notna(row.get('experimental_protein', None))
        }
        minimize_and_rerank(
            out_dir=args.out_dir,
            protein_path_map=protein_path_map,
            smina_path=args.smina_path,
            gnina_path=args.gnina_path,
            n_workers=args.minimize_workers,
        )


if __name__ == "__main__":
    mp_method = "spawn"
    sharing_strategy = "file_system"
    logging.debug(f"Torch multiprocessing method: {mp_method}. Sharing strategy: {sharing_strategy}")

    current_method = torch.multiprocessing.get_start_method(allow_none=True)
    if current_method is None:
        torch.multiprocessing.set_start_method(mp_method)
    elif current_method != mp_method:
        logging.warning(f"Multiprocessing start method already set to {current_method}, not changing to {mp_method}")

    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    parser = _get_parser()
    _args = parser.parse_args()
    with torch.no_grad():
        main(_args)
