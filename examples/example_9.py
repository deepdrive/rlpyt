
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import cv2


############################# classes and functions #############################

class CustomMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class CustomDqnModel(torch.nn.Module):
    def __init__(
            self,
            in_channels=3,
            n_actions=6,
            ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, observation, prev_action, prev_reward):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        x = img.view(T * B, *img_shape[::-1])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        q = self.head(x)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


# class CustomDqnAgent(CustomMixin, DqnAgent):
class CustomDqnAgent(DqnAgent):
    def __init__(self, ModelCls=CustomDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


def make_env_custom(*args, **kwargs):
    env = gym.make('Pong-v0')
    # env = make_env(env)
    env = ResizeFrame(env)
    env = GymEnvWrapper(env)
    return env


def build_and_train(run_ID=0, cuda_idx=None):
    # env_id = 'CartPole-v0'
    env_id = 'Pong-v0'

    sampler = AsyncCpuSampler(
        EnvCls=make_env_custom,
        env_kwargs=dict(id=env_id), #env_config,
        eval_env_kwargs=dict(id=env_id),  #env_config,
        batch_T=5,  # One time-step per sampler iteration.
        batch_B=8,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=100,
        eval_n_envs=2,
        eval_max_steps=int(10e3),
        eval_max_trajectories=4,
    )

    algo = DQN(
        replay_ratio = 8,
        min_steps_learn = 1e4,
        replay_size = int(1e5)
    )

    agent = CustomDqnAgent()

    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=3,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        sample_gpu_per_run=0,
        async_sample=True,
        # hyperthread_offset=24,  # If machine has 24 cores.
        # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=2e6,
        log_interval_steps=1e3,
        affinity=affinity #dict(cuda_idx=cuda_idx, workers_cpus=[0,1,2,4,5,6])
    )

    config = dict(env_id=env_id)
    algo_name = 'dqn_'
    name = algo_name + env_id
    log_dir = algo_name + env_id

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()

    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
