
import sys
sys.path.append('/home/isaac/codes/dd-zero/deepdrive-zero/')
sys.path.append('/home/isaac/codes/dd-zero/rlpyt/')

from deepdrive_zero.envs.env import Deepdrive2DEnv
import torch
import numpy as np

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector, CpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.envs.base import EnvSpaces


config = dict(
    agent=dict(),
    # https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
    algo=dict(
        discount=0.99,
        learning_rate=5e-5,
        clip_grad_norm=0.8, # higher -> suddenly collapse
        entropy_loss_coeff=0.005, # 1e-2-1e-4 -> higher: more exploration
        gae_lambda=0.94, #0.9- 0.95 -> higher: more variance, lower: more bias
        minibatches=4, #8 -> hint: minibatched should be less than batch_B -> I think in lstm version
        epochs=4, #4
        ratio_clip=0.15,
        normalize_advantage=False,
        linear_lr_schedule=False
    ),
    env = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=0, #3.3e-6 * 2,
        gforce_penalty_coeff=0, #0.006,
        lane_penalty_coeff=0.02, #0.02,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        gforce_threshold=None, ##question
        end_on_harmful_gs=False,
        incent_win=True, # reward for reaching the target
        incent_yield_to_oncoming_traffic=True, #this just consider env.agents not dummy ones. So doesn't have any effect on dummy opp agent
        constrain_controls=False,
        physics_steps_per_observation=6,
        contain_prev_actions_in_obs=False,
        dummy_accel_agent_indices=[1], #for opponent
        dummy_random_scenario=False, #select randomly between 3 scenarios for dummy agent
    ),
    model=dict(
        hidden_sizes=[256, 256],
        lstm_size=256,
        normalize_observation=False,
    ),
    optim=dict(),
    runner=dict(
        n_steps=3.5e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=128,
        batch_B=32,
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


def build_and_train(run_ID=0, cuda_idx=0, pre_trained_model=None):
    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    if pre_trained_model is not None:
        print('Continue from previous checkpoint ...')
        data = torch.load(pre_trained_model)
        agent_state_dict = data['agent_state_dict']
        optimizer_state_dict = data['optimizer_state_dict']
    else:
        print('start training from scratch ...')
        agent_state_dict = None
        optimizer_state_dict = None

    sampler = CpuSampler(
        EnvCls=make_env,
        env_kwargs=config["env"],
        eval_env_kwargs=config["env"],
        CollectorCls=CpuWaitResetCollector, #cuz of lstm, WaitReset collector is suggested by astooke
        **config["sampler"]
    )
    algo = PPO(
        initial_optim_state_dict=optimizer_state_dict,
        optim_kwargs=config["optim"],
        **config["algo"]
    )
    # agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    agent = MujocoLstmAgent(
        initial_model_state_dict=agent_state_dict,
        model_kwargs=config["model"],
        **config["agent"]
    )
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=[0, 1, 2, 3, 4, 5, 6])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )

    cfg = dict(env_id=config['env']['id'], **config)
    algo_name = 'ppo_lstm_mbopp_'
    if config['env']['dummy_random_scenario']:
        algo_name += '3scenario_'
    name = algo_name + config['env']['id']
    log_dir = algo_name + "dd0"

    with logger_context(log_dir, run_ID, name, cfg, snapshot_mode='last'):
        runner.train()


def evaluate(pre_trained_model):
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = Deepdrive2DEnv()
    env.configure_env(config['env'])

    agent = MujocoLstmAgent(
        initial_model_state_dict=agent_state_dict,
        model_kwargs=config["model"],
        **config["agent"]
    )
    env_spaces = EnvSpaces(
            observation=env.observation_space,
            action=env.action_space,
    )
    agent.initialize(env_spaces)
    # agent.sample_mode(0)

    obs = env.reset()
    prev_action = torch.tensor([0, 0, 0], dtype=torch.float)  # None
    prev_reward = torch.tensor(0.0, dtype=torch.float)  # None
    while True:
        action = agent.step(torch.tensor(obs, dtype=torch.float32), prev_action, prev_reward)
        action = np.array(action.action)
        # action = np.array([0,0,0])
        obs, reward, done, info = env.step(action)
        prev_action = torch.tensor(action, dtype=torch.float)
        prev_reward = torch.tensor(reward, dtype=torch.float)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--mode', help='train or eval', default='train')
    parser.add_argument('--pre_trained_model',
                        help='path to the pre-trained model.',
                        default='/home/isaac/codes/dd-zero/rlpyt/data/local/2020_04-15_11-01.22/ppo_lstm_mbopp_dd0_3rdscenario/run_0/params.pkl')

    args = parser.parse_args()

    if args.mode == 'train':
        build_and_train(
            run_ID=args.run_ID,
            cuda_idx=args.cuda_idx,
            pre_trained_model=None #args.pre_trained_model
        )
    else:
        evaluate(args.pre_trained_model)
