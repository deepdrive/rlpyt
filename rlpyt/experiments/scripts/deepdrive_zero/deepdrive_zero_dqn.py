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
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.deepdrive.deepdrive_dqn_agent import DeepDriveDqnAgent
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent, CategoricalEpsilonGreedy
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper, GymSpaceWrapper
from rlpyt.envs.base import EnvSpaces
from rlpyt.spaces.int_box import IntBox

import torch
import numpy as np

from gym import ActionWrapper
import gym

# env_config = dict(
#     id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
#     is_intersection_map=True,
#     jerk_penalty_coeff=0.10,
#     gforce_penalty_coeff=0.031,
#     lane_penalty_coeff=0.02,
#     collision_penalty_coeff=0.31,
#     speed_reward_coeff=0.50,
#     win_coefficient=1,
#     end_on_harmful_gs=True,
#     constrain_controls=True,
#     ignore_brake=False,
#     forbid_deceleration=False,
#     expect_normalized_action_deltas=True,
#     incent_win=True,
#     dummy_accel_agent_indices=[1],
#     wait_for_action=False,
#     incent_yield_to_oncoming_traffic=True,
# )

env_config = dict(
    id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    # id = 'deepdrive-2d-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0.1,
    gforce_penalty_coeff=0.01,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    dummy_accel_agent_indices=[1]
)



class DiscretizedActionWrapper(ActionWrapper):
    """ Discretizes the action space of an `env` using
        `transform.discretize()`.
        The `reverse_action` method is currently not implemented.
    """
    def __init__(self, env):
        super(DiscretizedActionWrapper, self).__init__(env)
        discrete_acc = [-1.0, 0.0, 0.5, 1.0]
        discrete_steer = [-0.2, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.2]
        self.discrete_act = [discrete_acc, discrete_steer]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        self.action_space = gym.spaces.Discrete(self.n_acc * self.n_steer)
        # self.action_space = IntBox(low=0, high=self.n_acc * self.n_steer, shape=(self.n_acc*self.n_steer,))

    def step(self, action):
        # action input is continues:
        # **steer**
        # > Heading angle of the ego
        #
        # **accel**
        # > m/s/s of the ego, positive for forward, negative for reverse
        #
        # **brake**
        # > From 0g at -1 to 1g at 1 of brake force

        acc = self.discrete_act[0][action // self.n_steer]
        steer = self.discrete_act[1][action % self.n_steer]

        if acc > 0:
            accel = acc
            brake = 0
        else:
            accel = 0
            brake = -acc

        act = np.array([steer, accel, brake])
        return self.env.step(act)


def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    env = DiscretizedActionWrapper(env)
    # return env
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

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    algo = DQN(
        batch_size=64,
        replay_size=100000,
        # bootstrap_timelimit=False,
    )
    agent = DeepDriveDqnAgent()

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=[0,1,2,3,4,5,6]),
    )

    config = dict(env_id=env_config['id'])
    name = "dqn_" + env_config['id']
    log_dir = "dd2d"

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


def evaluate():
    pre_trained_model = '/home/isaac/codes/dd-zero/rlpyt/data/local/2020_03-23_20-18.59/dd2d/run_0/params.pkl'
    data = torch.load(pre_trained_model)
    agent_state_dict = data['agent_state_dict']

    # for loading pre-trained models see: https://github.com/astooke/rlpyt/issues/69
    env = Deepdrive2DEnv()
    env.configure_env(env_config)
    env = DiscretizedActionWrapper(env)

    agent = Deepdrive2DEnv()
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
    args = parser.parse_args()

    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )

    # evaluate()

