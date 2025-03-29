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

def random_quaternion(env_ids: Tensor, *, rng: BatchedRNG):
    """Sample uniform random quaternions using Marsaglia method."""
    u1 = torch.from_numpy(rng.rand())[env_ids].to(torch.float32)
    u2 = torch.from_numpy(rng.rand())[env_ids].to(torch.float32)
    u3 = torch.from_numpy(rng.rand())[env_ids].to(torch.float32)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    return torch.stack([q1, q2, q3, q4], dim = -1)

def sample_rotations(env_ids: Tensor, rotations_pool: Tensor, *, rng: BatchedRNG):
    rotation_idxs = rng.randint(len(rotations_pool))[env_ids]
    return rotations_pool[rotation_idxs]

def unique_cube_rotations_3d() -> list[ndarray]:
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

    return torch.from_numpy(np.array(all_rotations)).to(torch.float32)
    
def batched_randint(lo: int, hi: int, *, dtype: torch.dtype, device: torch.device, rng: BatchedRNG):
    return torch.from_numpy(rng.randint(lo, hi)).to(dtype=dtype, device=device)