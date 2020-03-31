"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.
"""


from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.dqn.deepdrive.deepdrive_dqn_agent import DeepDriveDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.envs.gym import make as gym_make
from rlpyt.models.dqn.deepdrive_dqn_model import DeepDriveDqnModel
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer

import torch
import numpy as np
import gym


def build_and_train(run_ID=0, cuda_idx=None, resume_chkpnt=None):
    env_id = 'CartPole-v0'

    sampler = CpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id), #env_config,
        eval_env_kwargs=dict(id=env_id),  #env_config,
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=100,
        eval_n_envs=10,
        eval_max_steps=int(50e3),
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
        discount=0.98,
        batch_size=32,
        min_steps_learn=32,
        # delta_clip=None, # selects the Huber loss; if ``None``, uses MSE.
        replay_size=int(50e3),
        replay_ratio=8,  # data_consumption / data_generation.
        # target_update_tau=1,
        target_update_interval=20,  # 312 * 32 = 1e4 env steps.
        # n_step_return=1,
        learning_rate=0.001,
        # OptimCls=torch.optim.Adam,
        # optim_kwargs=None,
        # initial_optim_state_dict=optimizer_state_dict,
        # clip_grad_norm=2.,
        eps_steps=int(2222), #1e6  # STILL IN ALGO (to convert to itr).
        double_dqn=True,
        # prioritized_replay=False,
        pri_alpha=0.1,
        # pri_beta_init=0.4,
        # pri_beta_final=1.,
        pri_beta_steps=int(1e3),
        # default_priority=None,
        ReplayBufferCls=UniformReplayBuffer,  # Leave None to select by above options.
        # updates_per_sync=1,  # For async mode only.
    )

    agent = DeepDriveDqnAgent()
    # agent = DqnAgent()
    # agent.MocelCls = DeepDriveDqnModel

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=[0,1,2,3,4,5])
    )

    config = dict(env_id=env_id)
    algo_name = 'dqn_'
    name = algo_name + env_id
    log_dir = algo_name + "cartpole"

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def evaluate(resume_chkpnt):
    import time

    pre_trained_model = resume_chkpnt
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']


    env = gym.make('CartPole-v0')

    agent = DqnAgent()
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)
    agent.load_state_dict(agent_state_dict['model'])

    obs = env.reset()

    tot_reward = 0
    while True:
        action = agent.step(torch.tensor(obs, dtype=torch.float32), torch.tensor(0), torch.tensor(0))
        a = np.array(action.action)
        obs, reward, done, info = env.step(a)
        tot_reward += reward
        env.render()
        time.sleep(0.01)

        if done:
            break

    print('reward: ', tot_reward)
    env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--resume_chkpnt', help='set path to pre-trained model', type=str,
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_03-30_08-03.21/dqn_dd0/run_0/params.pkl')
    parser.add_argument('--no-timeout', help='consider timeout or not ', default=True)

    args = parser.parse_args()

    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        resume_chkpnt=None,
    )

    # evaluate(args.resume_chkpnt)