import sys

from deepdrive_zero.envs.env import Deepdrive2DEnv

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector

from rlpyt.algos.dqn.r2d1 import R2D1
from rlpyt.agents.dqn.deepdrive.deepdrive_r2d1_agent import DeepDriveR2d1Agent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.deepdrive_zero.dqn.dd0_r2d1_configs import configs

from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces
from rlpyt.utils.wrappers import DeepDriveDiscretizeActionWrapper


import torch
import numpy as np


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    env = DeepDriveDiscretizeActionWrapper(env)
    env = GymEnvWrapper(env)
    return env


def build_and_train(run_ID=0):

    affinity = dict(cuda_idx=0, workers_cpus=[0,1,2,3,4,5,6])
    config = configs['r2d1']

    cfg = dict(env_id=config['env']['id'])
    algo_name = 'r2d1_'
    name = algo_name + config['env']['id']
    log_dir = algo_name + "dd0"

    # TODO: doesn't work with CpuSampler. Check why?
    sampler = GpuSampler(
        EnvCls=make_env,
        env_kwargs=config['env'],
        CollectorCls=GpuWaitResetCollector,
        eval_env_kwargs=config['eval_env'],
        **config["sampler"]
    )

    algo = R2D1(optim_kwargs=config["optim"], **config["algo"])

    agent = DeepDriveR2d1Agent(**config["agent"])

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )

    with logger_context(log_dir, run_ID, name, cfg, snapshot_mode='last'):
        runner.train()


def evaluate(pre_trained_model):
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    config = configs['r2d1']
    env_config = config['eval_env']
    env = Deepdrive2DEnv()
    env.configure_env(env_config)
    env = DeepDriveDiscretizeActionWrapper(env)

    agent = DeepDriveR2d1Agent(initial_model_state_dict=agent_state_dict['model'])
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)

    obs = env.reset()
    while True:
        action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), torch.tensor(0), torch.tensor(0))
        a = np.array(action)
        obs, reward, done, info = env.step(a)
        env.render()
        if done:
            obs = env.reset()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help='train or eval', default='train')
    parser.add_argument('--pre_trained_model',
                        help='path to the pre-trained model.',
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_04-03_22-27.20/r2d1_dd0/run_0/params.pkl'
                        )

    args = parser.parse_args()

    if args.mode == 'train':
        build_and_train()
    else:
        evaluate(args.pre_trained_model)

