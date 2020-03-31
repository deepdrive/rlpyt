from deepdrive_zero.envs.env import Deepdrive2DEnv

import numpy as np

from rlpyt.envs.gym import DeepDriveDiscretizeActionWrapper

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env_config = dict(
    id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    is_one_waypoint_map=False,
    expect_normalized_actions=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0.0,
    gforce_penalty_coeff=0.0,
    lane_penalty_coeff=0.02,
    collision_penalty_coeff=0.31,
    speed_reward_coeff=0.50,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
    physics_steps_per_observation=6,
)

env = Deepdrive2DEnv()
env.configure_env(env_config)
env = DeepDriveDiscretizeActionWrapper(env)

def train():
    # baseline_dqn_dd0: layers: [128, 128], env action space: steer=[-1, 1, step=0.2], throttle=[-1, 0, 1], tensorboard: DQN_2
    model = DQN(MlpPolicy, env, policy_kwargs=dict(layers=[128, 128]), verbose=1, tensorboard_log="./dqn_dd0_tensorboard/")

    model.learn(total_timesteps=int(2e5))
    model.save("baseline_dqn_dd0_2")

def test():
    model = DQN.load("baseline_dqn_dd0_2")
    for _ in range(5):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            if dones:
                break


if __name__ == '__main__':
    # train()
    test()