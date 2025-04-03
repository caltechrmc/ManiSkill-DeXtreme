from __future__ import annotations

from scipy.spatial.transform import Rotation as R

import numpy as np
import torch

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
    from torch import Tensor
    from numpy import ndarray

def random_quaternion(env_ids: Tensor, *, rng: BatchedRNG) -> Tensor:
    """Sample uniform random quaternions using Marsaglia method."""
    u1 = torch.from_numpy(rng.rand())[env_ids].to(torch.float)
    u2 = torch.from_numpy(rng.rand())[env_ids].to(torch.float)
    u3 = torch.from_numpy(rng.rand())[env_ids].to(torch.float)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    return torch.stack([q1, q2, q3, q4], dim = -1)

def sample_rotations(env_ids: Tensor, rotations_pool: Tensor, *, rng: BatchedRNG) -> Tensor:
    random_integers = batched_randint(0, len(rotations_pool), dtype=torch.long, device=rotations_pool.device, rng=rng)
    rotation_idxs = random_integers[env_ids]
    return rotations_pool[rotation_idxs]

def unique_cube_rotations_3d() -> Tensor:
    """
    Returns the list of all possible 90-degree cube rotations in 3D.
    Based on https://stackoverflow.com/a/70413438/1645784
    """
    all_rotations = []
    for x, y, z in itertools.permutations([0, 1, 2]):
        for sx, sy, sz in itertools.product([-1, 1], repeat=3):
            rotation_matrix = np.zeros((3, 3))
            rotation_matrix[0, x] = sx
            rotation_matrix[1, y] = sy
            rotation_matrix[2, z] = sz
            
            if np.linalg.det(rotation_matrix) == 1:
                all_rotations.append(R.from_matrix(rotation_matrix).as_quat())

    return torch.from_numpy(np.array(all_rotations)).to(dtype=torch.float)

def quaternion_conjugate(q):
    """Compute the batched quaternion conjugate."""
    return torch.cat((q[:, :1], -q[:, 1:]), dim=1)

def quaternion_multiply(q1, q2):
    """Compute batched quaternion multiplications."""
    w1, xyz1 = q1[:, :1], q1[:, 1:]
    w2, xyz2 = q2[:, :1], q2[:, 1:]

    w = w1 * w2 - torch.sum(xyz1 * xyz2, dim=1, keepdim=True)
    xyz = w1 * xyz2 + w2 * xyz1 + torch.cross(xyz1, xyz2, dim=1)

    return torch.cat((w, xyz), dim=1)

def quaternion_difference(q1, q2):
    """Compute batched quaternion differences."""
    return quaternion_multiply(q1, quaternion_conjugate(q2))

def batched_randint(lo: int, hi: int, *, dtype: torch.dtype, device: torch.device, rng: BatchedRNG) -> Tensor:
    return torch.from_numpy(rng.randint(lo, hi)).to(dtype=dtype, device=device)