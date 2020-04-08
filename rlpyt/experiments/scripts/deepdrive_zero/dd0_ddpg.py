"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import sys
sys.path.append('/home/isaac/codes/dd-zero/deepdrive-zero')
sys.path.append('/home/isaac/codes/dd-zero/rlpyt')

from deepdrive_zero.envs.env import Deepdrive2DEnv
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces

import torch
import numpy as np


env_config = dict(
    id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0.1,
    gforce_penalty_coeff=0.01,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    dummy_accel_agent_indices=[1]
)


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    return GymEnvWrapper(env)


def build_and_train(run_ID=0, cuda_idx=None):
    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=env_config,
        eval_env_kwargs=env_config,
        batch_T=4,  # One time-step per sampler iteration.
        batch_B=8,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )


    algo = DDPG(
        batch_size=64,
        replay_size=100000,
        bootstrap_timelimit=False,
    )
    agent = DdpgAgent()

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=10,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=[0,1,2,3,4,5,6]),
    )

    config = dict(env_id=env_config['id'])
    name = "ddpg_" + env_config['id']
    log_dir = "dd2d"

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def evaluate(pre_trained_model):
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = Deepdrive2DEnv()
    env.configure_env(env_config)

    agent = DdpgAgent()
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)
    agent.load_state_dict(agent_state_dict)

    obs = env.reset()
    while True:
        action = agent.step(torch.tensor(obs, dtype=torch.float32), None, None)
        a = np.array(action.action)
        obs, reward, done, info = env.step(a)
        env.render()
        if done:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--mode', help='train or eval', default='train')
    parser.add_argument('--pre_trained_model',
                        help='path to the pre-trained model.',
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_03-23_20-18.59/dd2d/run_0/params.pkl')
    args = parser.parse_args()

    if args.mode == 'train':
        build_and_train(
            run_ID=args.run_ID,
            cuda_idx=args.cuda_idx
        )
    else:
        evaluate(args.pre_trained_model)

