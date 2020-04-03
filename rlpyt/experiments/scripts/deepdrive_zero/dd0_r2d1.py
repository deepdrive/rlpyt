import sys

from deepdrive_zero.envs.env import Deepdrive2DEnv

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
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

    algo_name = 'r2d1_'
    name = algo_name + config['env']['id']
    log_dir = algo_name + "dd0"

    # variant = load_variant(log_dir)
    # config = update_config(config, variant)

    sampler = GpuSampler(
        EnvCls=make_env,
        env_kwargs=config['env'],
        CollectorCls=GpuWaitResetCollector,
        eval_env_kwargs=config['eval_env'],
        **config["sampler"]
    )

    algo = R2D1(optim_kwargs=config["optim"], **config["algo"])

    agent = DeepDriveR2d1Agent(model_kwargs=config["model"], **config["agent"])

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )

    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train()
