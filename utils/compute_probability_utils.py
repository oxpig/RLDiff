import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import RemoveHs
from scipy.spatial.transform import Rotation as R
import os

def compute_translation_likelihood(action: torch.Tensor,
                                mean: torch.Tensor,
                                std_dev: torch.Tensor) -> torch.Tensor:
   """
   Log-prob of 'action' ~ Normal(mean, std_dev) in R^3.
   Args:
       action: shape [B, 3] - sampled translation
       mean: shape [B, 3] - mean translation
       std_dev: shape [B, 3] - standard deviation
   Returns:
       log_prob: shape [B] - log probability of each translation
   """
   epsilon = 1e-12
   std_dev = std_dev + epsilon
   
   # Compute log probability: -0.5*((x - mu)/sigma)^2 for each dimension
   log_prob = -0.5 * ((action - mean) / std_dev)**2
   log_prob = log_prob.sum(dim=-1)
   
   # Add normalization terms
   log_prob -= torch.log(std_dev).sum(dim=-1)
   log_prob -= 1.5 * math.log(2 * math.pi)
   return log_prob




def compute_rotation_likelihood(action: torch.Tensor,
                                mean: torch.Tensor,
                                std_dev: torch.Tensor) -> torch.Tensor:
   """
   Log-prob of 'action' ~ Normal(mean, std_dev) in R^3.
   Args:
       action: shape [B, 3] - sampled translation
       mean: shape [B, 3] - mean translation
       std_dev: shape [B, 3] - standard deviation
   Returns:
       log_prob: shape [B] - log probability of each translation
   """
   epsilon = 1e-12
   std_dev = std_dev + epsilon

   # Compute log probability: -0.5*((x - mu)/sigma)^2 for each dimension
   log_prob = -0.5 * ((action - mean) / std_dev)**2
   log_prob = log_prob.sum(dim=-1)

   # Add normalization terms
   log_prob -= torch.log(std_dev).sum(dim=-1)
   log_prob -= 1.5 * math.log(2 * math.pi)
   return log_prob




def compute_torsion_likelihood(
    action: torch.Tensor,
    mean: torch.Tensor,
    std_dev: torch.Tensor
) -> torch.Tensor:
    """
    Compute likelihood of axis–angle vectors (Gaussian in R^3).
    Args:
        action: [B, 3] - sampled axis–angle vectors
        mean:   [B, 3] - mean axis–angle vectors
        std_dev:[B, 3] - standard deviation
    """
    delta = action - mean
    exponent = -0.5 * (delta / std_dev) ** 2
    log_term = exponent - torch.log(std_dev * math.sqrt(2.0 * math.pi))
    return log_term.sum(dim=-1)   # sum over the 3 coords



def get_ground_truth_ligand(file_path: str, device: torch.device) -> torch.Tensor:
    """
    Load ligand coords in the SAME way as get_lig_graph_with_matching():
      SDMolSupplier(sanitize=False, removeHs=False) then RemoveHs(..., sanitize=True)

    This matches atom count + ordering used in the graph.
    """
    supplier = Chem.SDMolSupplier(file_path, sanitize=False, removeHs=False)
    if not supplier or supplier[0] is None:
        raise ValueError(f"Failed to load molecule from {file_path}")

    mol = supplier[0]

    # EXACTLY like preprocessing
    mol = RemoveHs(mol, sanitize=True)

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule from {file_path} has no conformers after RemoveHs.")

    conf = mol.GetConformer(0)
    coords = conf.GetPositions().astype(np.float32)  # (N,3)

    return torch.tensor(coords, device=device, dtype=torch.float32)


def kabsch_alignment(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(np.dot(Vt.T, U.T))
    D = np.eye(3)
    if d < 0:
        D[2, 2] = -1
    R_opt = np.dot(Vt.T, np.dot(D, U.T))
    return R_opt

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )

    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def create_rdkit_conformer(coords: np.ndarray) -> Chem.Conformer:
    """Create an RDKit Conformer from a NumPy array of coordinates."""
    num_atoms = coords.shape[0]
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        conf.SetAtomPosition(i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
    return conf


def build_adjacency_list(edge_index: torch.Tensor, num_atoms: int) -> dict:
    adj = {i: set() for i in range(num_atoms)}
    edges = edge_index.cpu().numpy()
    for u, v in edges.T:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj

def get_dihedral_angle_np(coords, a, u, v, d):
    """
    Compute dihedral angle (in radians) for atoms (a-u-v-d)
    given `coords` of shape (num_atoms, 3) in NumPy.

    Returns angle in (-pi, +pi].
    """
    p_a = coords[a]
    p_u = coords[u]
    p_v = coords[v]
    p_d = coords[d]

    b1 = p_a - p_u
    b2 = p_v - p_u
    b3 = p_d - p_v

    # Normal vectors to the planes (a,u,v) and (u,v,d)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Magnitudes
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0  # degenerate

    # Normalize plane normals
    n1 /= n1_norm
    n2 /= n2_norm

    # Angle between plane normals
    cos_angle = np.dot(n1, n2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # clamp
    angle = math.acos(cos_angle)  # in [0, π]

    # Sign determination.
    sign = np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2))
    if sign < 0:
        angle = -angle

    return angle  # in (-pi, +pi]


def compute_optimal_torsions(complex_graph: dict, gt_coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute optimal torsion updates on the original coordinates.
    """
    # Use the original coordinates from the graph (do not modify the graph!)
    mask_rotate = complex_graph['ligand'].mask_rotate
    if isinstance(mask_rotate, torch.Tensor):
        mask_rotate = mask_rotate.cpu().numpy()
    mask_rotate_squeezed = np.squeeze(mask_rotate, axis=(0, 1))
    num_rot_bonds, num_atoms = mask_rotate_squeezed.shape






    # (1) Start with the original coordinates (as a NumPy array)
    # Clone to avoid in-place modifications affecting the original tensor.
    current_coords = complex_graph['ligand'].pos.to(device).clone().cpu().numpy()
    pos = current_coords.copy()  # this copy will be modified in-place

    # Ground-truth coordinates in NumPy
    gt_coords_np = gt_coords.cpu().numpy()

    # Build adjacency list & identify rotatable edges.
    edge_index = complex_graph['ligand', 'ligand'].edge_index
    adj = build_adjacency_list(edge_index, num_atoms)

    edge_mask = complex_graph['ligand'].edge_mask
    if isinstance(edge_mask, torch.Tensor):
        edge_mask_np = edge_mask.cpu().numpy().astype(bool)
    else:
        edge_mask_np = np.array(edge_mask, dtype=bool)

    edges_np = edge_index.cpu().numpy().T
    rot_edges = edges_np[edge_mask_np]  # shape: [num_rot_bonds, 2]

    torsion_diffs = []
    for i, (u, v) in enumerate(rot_edges):
        # print(f"Rotatable bond {i}: Initial candidate edge: u={u}, v={v}")
        moving = set(np.where(mask_rotate_squeezed[i])[0])
        fixed = set(range(num_atoms)) - moving
        # print(f"Rotatable bond {i} - Moving indices: {sorted(moving)}")
        # print(f"Rotatable bond {i} - Fixed indices: {sorted(fixed)}")

        # Ensure proper orientation: u in fixed, v in moving.
        if (u in fixed and v in moving):
            pass
        elif (u in moving and v in fixed):
            u, v = v, u
            print(f"Rotatable bond {i} - Swapped edge order: u={u}, v={v}")
        else:
            print(f"Rotatable bond {i} - Edge ambiguous; skipping.")
            torsion_diffs.append(0.0)
            continue

        # Select neighbor atoms for the dihedral.
        fixed_neighbors = [x for x in adj[u] if x in fixed and x != v]
        if not fixed_neighbors:
            print(f"Rotatable bond {i} - No valid fixed neighbor for u={u}; skipping.")
            torsion_diffs.append(0.0)
            continue
        a = fixed_neighbors[0]

        moving_neighbors = [x for x in adj[v] if x in moving and x != u]
        if not moving_neighbors:
            print(f"Rotatable bond {i} - No valid moving neighbor for v={v}; skipping.")
            torsion_diffs.append(0.0)
            continue
        d = moving_neighbors[0]
        #print(f"Rotatable bond {i} - Selected dihedral: a={a}, u={u}, v={v}, d={d}")

        # (1) Compute current dihedral (from original coords)
        current_dihedral = get_dihedral_angle_np(pos, a, u, v, d)
        # (2) Compute ground-truth dihedral.
        gt_dihedral = get_dihedral_angle_np(gt_coords_np, a, u, v, d)
        # (3) Compute difference.
        diff = gt_dihedral - current_dihedral
        diff = -diff  # adjust sign if needed
        # Wrap into (-pi, +pi]
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff <= -math.pi:
            diff += 2 * math.pi

        # print(f"Rotatable bond {i} - Before rotation: Current dihedral={current_dihedral:.4f} rad, "
        #       f"GT dihedral={gt_dihedral:.4f} rad, diff={diff:.4f} rad")

        # (4) Inline rotation logic (compute rotation matrix and apply to moving atoms).
        rot_vec = pos[u] - pos[v]  # axis from v to u (or vice versa depending on sign)
        length = np.linalg.norm(rot_vec)
        if length < 1e-12:
            torsion_diffs.append(0.0)
            continue
        rot_vec = rot_vec * diff / length
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        # Rotate all moving atoms.
        moving_mask = mask_rotate_squeezed[i]  # Boolean mask for atoms that move.
        pos[moving_mask] = (pos[moving_mask] - pos[v]) @ rot_mat.T + pos[v]

        # Recompute dihedral after rotation.
        updated_dihedral = get_dihedral_angle_np(pos, a, u, v, d)
        #print(f"Rotatable bond {i} - After rotation: Updated dihedral={updated_dihedral:.4f} rad, GT dihedral={gt_dihedral:.4f} rad")

        torsion_diffs.append(diff)

        # Optionally print all dihedrals after each update.
        # print(f"\nAll dihedrals after updating bond {i}:")
        for j in range(num_rot_bonds):
            u_j, v_j = rot_edges[j]
            moving_j = set(np.where(mask_rotate_squeezed[j])[0])
            fixed_j = set(range(num_atoms)) - moving_j
            if (u_j in fixed_j and v_j in moving_j):
                pass
            elif (u_j in moving_j and v_j in fixed_j):
                u_j, v_j = v_j, u_j
            else:
                print(f"Bond {j}: ambiguous orientation, skipped.")
                continue
            fixed_neighbors_j = [x for x in adj[u_j] if x in fixed_j and x != v_j]
            if not fixed_neighbors_j:
                print(f"Bond {j}: no valid fixed neighbor, skipped.")
                continue
            a_j = fixed_neighbors_j[0]
            moving_neighbors_j = [x for x in adj[v_j] if x in moving_j and x != u_j]
            if not moving_neighbors_j:
                print(f"Bond {j}: no valid moving neighbor, skipped.")
                continue
            d_j = moving_neighbors_j[0]
            curr_dih = get_dihedral_angle_np(pos, a_j, u_j, v_j, d_j)
            gt_dih = get_dihedral_angle_np(gt_coords_np, a_j, u_j, v_j, d_j)
            # print(f"Bond {j}: Current dihedral = {curr_dih:.4f} rad, GT dihedral = {gt_dih:.4f} rad")
        # print("------------------------------------------------------\n")

    if len(torsion_diffs) == 0:
        return torch.tensor([], device=device, dtype=torch.float32)

    torsion_tensor = torch.tensor(torsion_diffs, device=device, dtype=torch.float32)
    #print("Final computed optimal torsions:", torsion_tensor)
    return torsion_tensor


def pick_closest_torsion(optimal_torsions, mean_tor_new):
    """
    Ensure each optimal torsion is the closest representation to mean_tor_new,
    resolving 2π periodicity.
    """
    adjusted = []
    for opt, mean in zip(optimal_torsions, mean_tor_new.squeeze(0)):
        candidates = torch.stack([
            opt,
            opt + 2 * math.pi,
            opt - 2 * math.pi,
            opt + 4 * math.pi,
            opt - 4 * math.pi
        ])
        diffs = torch.abs(candidates - mean)
        best_idx = torch.argmin(diffs)
        adjusted.append(candidates[best_idx])
    return torch.stack(adjusted)





def compute_optimal_score(complex_graph: dict,
                          device: torch.device,
                          args,
                          pre_scaled_mean_tr: int = None,
                          pre_scaled_mean_tor: int = None,
                          pre_scaled_mean_rot: int = None,
                          tr_cum: int = None, tor_cum: int = None, rot_cum: int = None,
                          mean_rot_new=None, mean_tor_new=None):
    pdb_id = complex_graph.name[0][0]
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.pdbbind_dir))
    ligand_file = f"{base_dir}/{pdb_id}/{pdb_id}_ligand.sdf"

    # Load ground truth coordinates (global)
    gt_coords = get_ground_truth_ligand(ligand_file, device=device)

    num_graph_atoms = complex_graph['ligand'].pos.shape[0]
    num_gt_atoms = gt_coords.shape[0]
    if num_graph_atoms != num_gt_atoms:
        raise ValueError(
            f"[{pdb_id}] Atom mismatch: graph has {num_graph_atoms}, GT has {num_gt_atoms}. "
            f"Check SDF loading / RemoveHs / sanitization."
        )

    # (Step 1) Capture the original (relative) coordinates from the graph.
    original_coords = complex_graph['ligand'].pos.to(device).clone()

    # Get the stored original center (global offset)
    original_center = complex_graph.original_center.to(device)

    # Convert the relative coordinates to global coordinates.
    original_coords_global = original_coords + original_center

    # (Step 1) Compute optimal torsion updates using the original (relative) coordinates.
    # (Assuming torsions are computed in the same coordinate frame)
    tor_score_optimal = compute_optimal_torsions(complex_graph, gt_coords, device)
    tor_score_optimal = pick_closest_torsion(tor_score_optimal, mean_tor_new)


    # Prepare inputs for applying torsion updates.
    batched_coords = original_coords.clone().unsqueeze(0)  # shape: [1, N, 3]

    # Prepare additional inputs.
    edge_index = complex_graph['ligand', 'ligand'].edge_index
    B = complex_graph.num_graphs
    N = complex_graph['ligand'].num_nodes // B
    M = complex_graph['ligand', 'ligand'].num_edges // B

    edge_index = complex_graph['ligand', 'ligand'].edge_index[:, :M]
    edge_mask = complex_graph['ligand'].edge_mask[:M]

    mask_rotate = torch.from_numpy(np.array(complex_graph['ligand'].mask_rotate[0])).to(device)
    mask_rotate = mask_rotate.squeeze(0)  # Shape: [num_rot_bonds, N]
    mask_rotate_np = mask_rotate.cpu().numpy()


    if isinstance(edge_mask, torch.Tensor):
        edge_mask_np = edge_mask.cpu().numpy().astype(bool)
    else:
        edge_mask_np = np.array(edge_mask, dtype=bool)
    edges_np = edge_index.cpu().numpy().T
    rot_edges = edges_np[edge_mask_np]  # shape: [num_rot_bonds, 2]
    adj = build_adjacency_list(edge_index, N)

    # (Step 2) Compute centers-of-geometry on the GLOBAL coordinates.
    current_cog = torch.mean(original_coords_global, dim=0, keepdim=True)
    gt_cog = torch.mean(gt_coords, dim=0, keepdim=True)

    # (Step 2) Compute optimal translation using GLOBAL coordinates.
    tr_score_optimal = gt_cog - current_cog

    # (Step 2) Compute optimal rotation using Kabsch on centered GLOBAL coordinates.
    P = (original_coords_global - current_cog).cpu().numpy()
    Q = (gt_coords - gt_cog).cpu().numpy()
    R_opt = kabsch_alignment(P, Q)
    R_opt = torch.tensor(R_opt, dtype=torch.float32, device=device)
    rot_score_optimal = matrix_to_axis_angle(R_opt).to(device)
    rot_score_optimal = rot_score_optimal.unsqueeze(0)

    if torch.norm(rot_score_optimal) < 1e-6:
        rot_score_optimal = torch.zeros_like(rot_score_optimal)

    rot_score_optimal_inverse = -rot_score_optimal
    d1 = torch.norm(rot_score_optimal - mean_rot_new)
    d2 = torch.norm(rot_score_optimal_inverse - mean_rot_new)
    rot_score_optimal = rot_score_optimal if d1 <= d2 else rot_score_optimal_inverse


    # # (Step 3) Apply the full transformation to the GLOBAL coordinates.
    # orig_pos_global = original_coords_global.clone().unsqueeze(0)  # shape: [1, N, 3]

    tr_mean_optimal = tr_score_optimal * (pre_scaled_mean_tr  / tr_cum)
    rot_mean_optimal = rot_score_optimal * (pre_scaled_mean_rot / rot_cum)
    tor_mean_optimal = tor_score_optimal * (pre_scaled_mean_tor / tor_cum)

    return tr_mean_optimal, rot_mean_optimal, tor_mean_optimal
