"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.
"""

from deepdrive_zero.envs.env import Deepdrive2DEnv

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.deepdrive.deepdrive_dqn_agent import DeepDriveDqnAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces
from rlpyt.utils.wrappers import DeepDriveDiscretizeActionWrapper
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.utils.seed import set_seed
from rlpyt.utils.logging import logger

import torch
import numpy as np
import gym


env_config = dict(
    id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    # id='deepdrive-2d-one-waypoint-v0',
    is_intersection_map=True,
    is_one_waypoint_map=False,
    expect_normalized_actions=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0.0,
    gforce_penalty_coeff=0.0,
    lane_penalty_coeff=0.02, #0.02
    collision_penalty_coeff=0.31,
    speed_reward_coeff=0.50,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    physics_steps_per_observation=6,
)


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    env = DeepDriveDiscretizeActionWrapper(env)
    env = GymEnvWrapper(env)
    return env


def build_and_train(run_ID=0, cuda_idx=None):
    resume_chkpnt = None

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=env_config,
        eval_env_kwargs=env_config,
        batch_T=32,  # One time-step per sampler iteration.
        batch_B=64,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    if resume_chkpnt is not None:
        print('Continue from previous checkpoint ...')
        data = torch.load(resume_chkpnt)
        agent_state_dict = data['agent_state_dict']['model']
        optimizer_state_dict = data['optimizer_state_dict']
    else:
        print('start training from scratch ...')
        agent_state_dict = None
        optimizer_state_dict = None

    algo = DQN(
        learning_rate=5e-4,
        replay_ratio=8,
        batch_size=32,
        min_steps_learn=1e3,
        eps_steps=10e3,
        replay_size=int(5e4),
        double_dqn=True,
        target_update_interval=int(500), #20
        # prioritized_replay=True,
        ReplayBufferCls=UniformReplayBuffer,
    )

    agent = DeepDriveDqnAgent(eps_final=0.02)

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=[0, 1, 2, 3, 4, 5, 6])
    )

    config = dict(env_id=env_config['id'])
    algo_name = 'dqn_'
    name = algo_name + env_config['id']
    log_dir = algo_name + "dd0"

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def evaluate():
    pre_trained_model = '/home/isaac/codes/dd-zero/rlpyt/data/local/2020_04-02_22-04.55/dqn_dd0/run_0/params.pkl'
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = Deepdrive2DEnv()
    env.configure_env(env_config)
    env = DeepDriveDiscretizeActionWrapper(env)

    agent = DeepDriveDqnAgent(initial_model_state_dict=agent_state_dict['model'])
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )

    agent.initialize(env_spaces)

    obs = env.reset()
    while True:
        action = agent.step(torch.tensor(obs, dtype=torch.float32), torch.tensor(0), torch.tensor(0))
        a = np.array(action.action)
        # logger.log()
        obs, reward, done, info = env.step(a)
        env.render()
        if done:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--no-timeout', help='consider timeout or not ', default=True)

    args = parser.parse_args()

    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )

    # evaluate()
