import sys
sys.path.append('/home/isaac/codes/dd-zero/deepdrive-zero')
sys.path.append('/home/isaac/codes/dd-zero/rlpyt')


from deepdrive_zero.envs.env import Deepdrive2DEnv
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.algos.dqn.r2d1 import R2D1
from rlpyt.agents.dqn.deepdrive.deepdrive_r2d1_agent import DeepDriveR2d1Agent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.experiments.configs.deepdrive_zero.dqn.dd0_r2d1_configs import configs
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces
from rlpyt.utils.wrappers import DeepDriveDiscretizeActionWrapper

import torch
import numpy as np
import gym
import time

##########################################################3
config = dict(
    agent=dict(),
    model=dict(
        mlp_hidden_sizes=[64],
        fc_size=64,  # Between mlp and lstm.
        lstm_size=64,
        head_size=64,
        dueling=False),
    algo=dict(
        discount=0.997,
        batch_T=4,
        batch_B=8,  # In the paper, 64.
        warmup_T=4,
        store_rnn_state_interval=4,
        replay_ratio=8,  # In the paper, more like 0.8.
        replay_size=int(1e3),
        learning_rate=1e-3,
        clip_grad_norm=100,  # 80 (Steven.) #TODO:test sth like 1e6. same as mujoco ppo
        min_steps_learn=int(256),
        eps_steps=int(1e4),
        target_update_interval=100, #2500
        double_dqn=True,
        frame_state_space=False,
        prioritized_replay=False,
        input_priorities=False, ## True - set it false for non-prioritized replay
        n_step_return=5, #5 #in the prioritization formula, r2d1 uses n-step return td-error -> I think we have to use n_step if we want to use prioritized replay
        pri_alpha=0.9,  # Fixed on 20190813
        pri_beta_init=0.6,  # I think had these backwards before.
        pri_beta_final=0.6,
        replay_buffer_class=None, #UniformSequenceReplayBuffer,
        input_priority_shift=2,  # Added 20190826 (used to default to 1)
    ),
    optim=dict(),
    env = dict(
        id='CartPole-v0',
    ),
    runner=dict(
        n_steps=10e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=4,  # Match the algo / replay_ratio.
        batch_B=8,
        max_decorrelation_steps=100,
        eval_n_envs=2,
        eval_max_steps=int(10e3),
        eval_max_trajectories=4,
    ),
)


def make_env(*args, **kwargs):
    env = gym.make('CartPole-v0')
    env = GymEnvWrapper(env)
    return env


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

    affinity = dict(cuda_idx=0, workers_cpus=range(27))
    cfg = dict(env_id=config['env']['id'], **config)
    algo_name = 'r2d1_'
    name = algo_name + config['env']['id']
    log_dir = algo_name + "cartpole"

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=config['env'],
        CollectorCls=CpuWaitResetCollector, #is beneficial for lstm based methods -> https://rlpyt.readthedocs.io/en/latest/pages/collector.html#rlpyt.samplers.parallel.cpu.collectors.CpuWaitResetCollector
        eval_env_kwargs=config['env'],
        **config["sampler"]
    )

    algo = R2D1(
        initial_optim_state_dict=optimizer_state_dict,
        optim_kwargs=config["optim"],
        **config["algo"]
    )

    agent = DeepDriveR2d1Agent(
        initial_model_state_dict=agent_state_dict,
        **config["agent"]
    )

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
    prev_action = torch.tensor(0.0, dtype=torch.float) #None
    prev_reward = torch.tensor(0.0, dtype=torch.float) #None
    while True:
        #TODO: do we need warm-up for evaluation too?
        action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), prev_action, prev_reward)
        action = np.array(action.action)
        obs, reward, done, info = env.step(action)
        prev_action = torch.tensor(action, dtype=torch.float)
        prev_reward = torch.tensor(reward, dtype=torch.float)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', help='train or eval', default='train')
    parser.add_argument('--pre_trained_model',
                        help='path to the pre-trained model.',
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_04-13_22-27.59/r2d1_dd0/run_0/params.pkl'
                        )

    args = parser.parse_args()

    if args.mode == 'train':
        #build_and_train(pre_trained_model=args.pre_trained_model)
        build_and_train()
    else:
        evaluate(args.pre_trained_model)

