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
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.deepdrive.deepdrive_dqn_agent import DeepDriveDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces
from rlpyt.envs.gym import DeepDriveDiscretizeActionWrapper

import torch
import numpy as np


# env_config = dict(
#     id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
#     # id = 'deepdrive-2d-one-waypoint-v0',
#     is_intersection_map=True,
#     jerk_penalty_coeff=0, #0.10,
#     gforce_penalty_coeff=0, #0.031,
#     lane_penalty_coeff=0.05, #0.05
#     collision_penalty_coeff=0.31,
#     speed_reward_coeff=0.50,
#     win_coefficient=1,
#     end_on_harmful_gs=True,
#     constrain_controls=True,
#     ignore_brake=False,
#     forbid_deceleration=False,
#     expect_normalized_action_deltas=True,
#     incent_win=True,
#     # dummy_accel_agent_indices=[1],
#     wait_for_action=False,
#     incent_yield_to_oncoming_traffic=True,
#     physics_steps_per_observation=6
# )


env_config = dict(
    id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_actions = True,
    expect_normalized_action_deltas=True,
    jerk_penalty_coeff=0.0,
    gforce_penalty_coeff=0.0,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    physics_steps_per_observation=6,
)


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    env = DeepDriveDiscretizeActionWrapper(env)
    return GymEnvWrapper(env)


def build_and_train(run_ID=0, cuda_idx=None, resume_chkpnt=None):
    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=env_config,
        eval_env_kwargs=env_config,
        batch_T=4,  # One time-step per sampler iteration.
        batch_B=16,  # One environment (i.e. sampler Batch dimension).
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
        initial_optim_state_dict=optimizer_state_dict,
        min_steps_learn=int(1e3),
        n_step_return=1,
        delta_clip=.5,
        eps_steps=int(1e5),
        learning_rate=1e-4,
        target_update_tau=0.01,
        replay_ratio=32,
        clip_grad_norm=10,
        # prioritized_replay = True,
        double_dqn = True,
        batch_size = 64,
        replay_size = int(1e5),
    )

    agent = DeepDriveDqnAgent(initial_model_state_dict=agent_state_dict)

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=3e7,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=[0,1,2,3,4,5,6]),
    )

    config = dict(env_id=env_config['id'])
    algo_name = 'dqn_'
    name = algo_name + env_config['id']
    log_dir = algo_name + "ddzero"

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def evaluate(resume_chkpnt):
    pre_trained_model = resume_chkpnt
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = Deepdrive2DEnv()
    env.configure_env(env_config)
    env = DeepDriveDiscretizeActionWrapper(env)

    agent = DeepDriveDqnAgent()
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)
    agent.load_state_dict(agent_state_dict['model'])

    for i in range(20):
        obs = env.reset()
        while True:
            action = agent.step(torch.tensor(obs, dtype=torch.float32), None, None)
            a = np.array(action.action)
            obs, reward, done, info = env.step(a)
            env.render()
            if done:
                break


def test():
    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = Deepdrive2DEnv()
    env.configure_env(env_config)
    env = DeepDriveDiscretizeActionWrapper(env)

    for i in range(5):
        obs = env.reset()
        while True:
            # a = np.array([1])
            a = np.random.randint(0, env.action_space.n)
            obs, reward, done, info = env.step(a)
            env.render()
            if done:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--resume_chkpnt', help='set path to pre-trained model', type=str,
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_03-27_19-42.42/dqn_ddzero/run_0/params.pkl')
    parser.add_argument('--no-timeout', help='consider timeout or not ', default=True)

    args = parser.parse_args()

    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        resume_chkpnt=None,
    )

    # evaluate(args.resume_chkpnt)
    # test()