from tasks.reorient_cube import *

from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from mani_skill.utils.wrappers import RecordEpisode
from sb3_contrib import RecurrentPPO

import gymnasium as gym
import numpy as np
import torch

N = 64
env = gym.make("ReorientCube-v0", num_envs=N, obs_mode="state", render_mode="rgb_array", reward_mode="normalized_dense", device="cuda")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, trajectory_name="trajectory", save_video=True, video_fps=60, render_substeps=False)
env = ManiSkillSB3VectorEnv(env)

model = RecurrentPPO("MlpLstmPolicy", env,
    gamma = 0.998,
    clip_range = 0.2,
    verbose = 1,
    target_kl = 0.016,
    policy_kwargs = {
        "activation_fn": torch.nn.ELU,
        "lstm_hidden_size": 2048,
        "net_arch": dict(vf=[1024, 512], pi=[512, 512])
    },
    device = "cuda",
    tensorboard_log = "logs",
)

model.learn(total_timesteps=int(2e5), progress_bar=True)