
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.wrappers import *
from rlpyt.envs.gym import make as make_env
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.envs.base import EnvSpaces
from rlpyt.utils.buffer import buffer_to

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym



############################# classes and functions #############################

class CustomMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class CustomDqnModel(torch.nn.Module):
    def __init__(
            self,
            observation_shape,
            output_size,
            fc_sizes=64
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_shape = observation_shape[0]

        self.base_net = torch.nn.Sequential(
            torch.nn.Linear(input_shape, fc_sizes),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_sizes, fc_sizes),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_sizes, output_size),
        )
        # self.base_net.apply(self.init_weights)

    def forward(self, observation, prev_action, prev_reward):
        observation = observation.type(torch.float)
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, self._obs_ndim)
        obs = observation.view(T * B, -1)
        q = self.base_net(obs)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight)
            torch.nn.init.zeros_(m.bias)


# class CustomDqnAgent(CustomMixin, DqnAgent):
class CustomDqnAgent(CustomMixin, DqnAgent):
    def __init__(self, ModelCls=CustomDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    @torch.no_grad()
    def eval_step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy. (no grad)"""
        # prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q = self.model(*model_inputs)
        q = q.cpu()
        action = torch.argmax(q)
        return action


def make_env_custom(*args, **kwargs):
    env = gym.make('CartPole-v0')
    env = GymEnvWrapper(env)
    return env


def build_and_train(run_ID=0, cuda_idx=None):
    env_id = 'CartPole-v0'

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=dict(id=env_id), #env_config,
        eval_env_kwargs=dict(id=env_id),  #env_config,
        batch_T=4,
        batch_B=8,
        max_decorrelation_steps=100,
        eval_n_envs=2,
        eval_max_steps=int(10e3),
        eval_max_trajectories=4,
    )

    algo = DQN(
        learning_rate=1e-3,
        replay_ratio=8,
        batch_size=32,
        min_steps_learn=32,
        eps_steps=10e3,
        replay_size=int(1e3),
        # double_dqn=True,
        # target_update_interval=1,
        # prioritized_replay=True,
        frame_state_space=False,
    )

    agent = CustomDqnAgent()

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e2,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=[0, 1, 2, 4, 5, 6])
    )

    config = dict(env_id=env_id)
    algo_name = 'dqn_'
    name = algo_name + env_id
    log_dir = algo_name + env_id

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def evaluate():
    import time
    pre_trained_model = '/home/isaac/codes/dd-zero/rlpyt/data/local/2020_04-04_06-52.20/dqn_CartPole-v0/run_0/itr_24713.pkl'
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = gym.make('CartPole-v0')

    agent = CustomDqnAgent(initial_model_state_dict=agent_state_dict['model'])

    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)
    agent.load_state_dict(agent_state_dict['model'])

    obs = env.reset()
    tot_reward = 0
    while True:
        # action = agent.step(torch.tensor(obs, dtype=torch.float32), torch.tensor(0), torch.tensor(0))
        action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), None, None)
        a = np.array(action)
        obs, reward, done, info = env.step(a)
        tot_reward += reward
        env.render()
        time.sleep(0.001)
        if done:
            break

    print('reward: ', tot_reward)
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--mode', help='train or eval', default='train')

    args = parser.parse_args()

    if args.mode == 'train':
        build_and_train(
            run_ID=args.run_ID,
            cuda_idx=args.cuda_idx,
        )
    else:
        evaluate()
