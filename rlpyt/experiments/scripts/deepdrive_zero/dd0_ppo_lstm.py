
import sys

from deepdrive_zero.envs.env import Deepdrive2DEnv
import torch
import numpy as np

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector, CpuWaitResetCollector
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=1e-4,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=4,
        epochs=4,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=True
    ),
    env = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=3.3e-6,
        gforce_penalty_coeff=0.006,
        lane_penalty_coeff=0.1, #0.02,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        end_on_harmful_gs=False,
        incent_win=True,
        incent_yield_to_oncoming_traffic=True,
        constrain_controls=False,
        physics_steps_per_observation=6,
        contain_prev_actions_in_obs=False,
        dummy_accel_agent_indices=[1] #for opponent
    ),
    model=dict(
        hidden_sizes=[128, 128],
        lstm_size=128,
        normalize_observation=False,
    ),
    optim=dict(),
    runner=dict(
        n_steps=10e6,
        log_interval_steps=1e3,
    ),
    sampler=dict(
        batch_T=32,
        batch_B=64,
        eval_n_envs=2,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
        max_decorrelation_steps=0,
    ),
)


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    return GymEnvWrapper(env)


def build_and_train(pre_trained_model=None, run_ID=0):
    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    if pre_trained_model is not None:
        print('Continue from previous checkpoint ...')
        data = torch.load(pre_trained_model)
        agent_state_dict = data['agent_state_dict']['model']
        optimizer_state_dict = data['optimizer_state_dict']
    else:
        print('start training from scratch ...')
        agent_state_dict = None
        optimizer_state_dict = None

    affinity = dict(cuda_idx=0, workers_cpus=[0, 1, 2, 3, 4, 5, 6])

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=config["env"],
        eval_env_kwargs=config["env"],
        CollectorCls=CpuWaitResetCollector, #cuz of lstm, WaitReser collector is suggested by astooke
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    # agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    agent = MujocoLstmAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )

    cfg = dict(env_id=config['env']['id'], **config)
    algo_name = 'ppo_lstm_'
    name = algo_name + config['env']['id']
    log_dir = algo_name + "dd0"

    with logger_context(log_dir, run_ID, name, cfg, snapshot_mode='last'):
        runner.train()


if __name__ == "__main__":
    build_and_train()
