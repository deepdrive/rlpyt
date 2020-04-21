import sys
sys.path.append('/home/isaac/codes/dd-zero/deepdrive-zero')
sys.path.append('/home/isaac/codes/dd-zero/rlpyt')


from deepdrive_zero.envs.env import Deepdrive2DEnv
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.algos.qpg.r2dpg import R2DPG
from rlpyt.agents.qpg.deepdrive_r2dpg_agent import DeepDriveR2dpgAgent
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
    model_kwargs=dict(
        mlp_hidden_sizes=[256],
        fc_size=256,  # Between mlp and lstm.
        lstm_size=256,
        head_size=256,
    ),
    q_model_kwargs=dict(
        mlp_hidden_sizes=[256],
        fc_size=256,  # Between mlp and lstm.
        lstm_size=256,
        head_size=256,
    ),
    algo=dict(
        discount=0.997,
        batch_T=80,
        batch_B=64,  # In the paper, 64.
        warmup_T=20, #or 40
        store_rnn_state_interval=40, # this shows overlap between sequences that we sample
        replay_ratio=64,  # In the paper, more like 0.8.
        replay_size=int(1e5),
        learning_rate=5e-5,
        q_learning_rate=1e-4,
        clip_grad_norm=80,  # 80 (Steven.)
        q_target_clip=1e6,
        min_steps_learn=int(1e4),
        target_update_tau=0.1,
        target_update_interval=100,
        policy_update_interval=1, #don't change this
        frame_state_space=False,
        prioritized_replay=False,
        input_priorities=False, ## True
        n_step_return=5, #5 #in the prioritization formula, r2d1 uses n-step return td-error -> I think we have to use n_step if we want to use prioritized replay
        pri_alpha=0.6,  # Fixed on 20190813
        pri_beta_init=0.9,  # I think had these backwards before.
        pri_beta_final=0.9,
        replay_buffer_class=None, #UniformSequenceReplayBuffer,
        input_priority_shift=1,  # Added 20190826 (used to default to 1)
    ),
    optim=dict(),
    env = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=0, #3.3e-6,
        gforce_penalty_coeff=0, #0.006,
        lane_penalty_coeff=0.04, #0.02,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        gforce_threshold=None,
        end_on_harmful_gs=False,
        incent_win=True,
        incent_yield_to_oncoming_traffic=True,
        constrain_controls=False,
        physics_steps_per_observation=6,
        contain_prev_actions_in_obs=False,
        dummy_accel_agent_indices=[1] #for opponent
    ),
    runner=dict(
        n_steps=50e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=30,  # Match the algo / replay_ratio.
        batch_B=27,
        max_decorrelation_steps=100,
        eval_n_envs=2,
        eval_max_steps=int(51e3),
        eval_max_trajectories=10,
    ),
)


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
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
    algo_name = 'r2dpg_'
    name = algo_name + config['env']['id']
    log_dir = algo_name + "dd0"

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=config['env'],
        CollectorCls=CpuWaitResetCollector, #is beneficial for lstm based methods -> https://rlpyt.readthedocs.io/en/latest/pages/collector.html#rlpyt.samplers.parallel.cpu.collectors.CpuWaitResetCollector
        eval_env_kwargs=config['env'],
        **config["sampler"]
    )

    algo = R2DPG(
        initial_optim_state_dict=optimizer_state_dict,
        optim_kwargs=config["optim"],
        **config["algo"]
    )

    agent = DeepDriveR2dpgAgent(
        initial_model_state_dict=agent_state_dict,
        model_kwargs=config["model_kwargs"],
        q_model_kwargs=config["q_model_kwargs"],
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
    env_config = config['env']
    env = Deepdrive2DEnv()
    env.configure_env(env_config)

    agent = DeepDriveR2dpgAgent(
        # initial_model_state_dict=agent_state_dict,
        model_kwargs=config["model_kwargs"],
        q_model_kwargs=config["q_model_kwargs"],
        **config["agent"]
    )
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)
    agent.load_state_dict(agent_state_dict)
    # agent.sample_mode(0)

    obs = env.reset()
    prev_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float) #None
    prev_reward = torch.tensor(0.0, dtype=torch.float) #None
    while True:
        #TODO: do we need warm-up for evaluation too?
        action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), prev_action, prev_reward)
        action = np.array(action.action) #np.array(action.agent_info.mu)
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
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_04-20_13-05.30/r2dpg_dd0/run_0/params.pkl'
                        )

    args = parser.parse_args()

    if args.mode == 'train':
        # build_and_train(pre_trained_model=args.pre_trained_model)
        build_and_train()
    else:
        evaluate(args.pre_trained_model)

