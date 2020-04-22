
from rlpyt.envs.base import Env
from rlpyt.envs.gym import IntBox
import gym
import numpy as np
import random


class DeepDriveDiscretizeActionWrapper(gym.ActionWrapper, Env):
    """ Discretizes the action space of deepdrive_zero env.
    """
    def __init__(self, env):
        super(DeepDriveDiscretizeActionWrapper, self).__init__(env)
        discrete_steer = list(np.arange(-0., 0.21, 0.05)) #list(np.arange(-1, 1.01, 0.08))
        discrete_acc   = [-1, 0, 0.1, 0.5, 1] # list(np.arange(0, 1.01, 0.25))
        # discrete_brake =[0, 1] # list(np.arange(-1, 1.01, 0.5))
        self.discrete_act = [discrete_steer, discrete_acc]  # acc, steer
        self.n_steer = len(self.discrete_act[0])
        self.n_acc = len(self.discrete_act[1])
        # self.n_brake = len(self.discrete_act[2])
        self.action_space = gym.spaces.Discrete(self.n_steer * self.n_acc)
        # self.action_space = IntBox(low=0, high=self.n_acc * self.n_steer)

        self.action_items = []
        for s in discrete_steer:
            for a in discrete_acc:
                # for b in discrete_brake:
                #     self.action_items.append([s, a, b])
                if a >= 0:
                   self.action_items.append([s, a, -0.5])
                else:
                   self.action_items.append([s, 0, -a])

        ##
        # self.prev_steer = 0
        # self.prev_accel = 0
        # self.prev_brake = 0
        #
        # steer_diff = [-0.05, 0, 0.05]
        # accel_diff = [-0.1, 0, 0.1]
        # # brake_diff = [-0.1, 0, 0.1]
        #
        # self.act_diff = []
        # for s in steer_diff:
        #     for a in accel_diff:
        #         self.act_diff.append([s, a])
        #
        # self.action_space = gym.spaces.Discrete(len(self.act_diff))

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
        # [steer, accel, brake]

        act = self.action_items[action]

        # curr_steer = self.prev_steer + self.act_diff[action][0]
        # curr_accel = self.prev_accel + self.act_diff[action][1]
        #
        # if curr_accel > 0:
        #     act = np.array([curr_steer, curr_accel, 0])
        # else:
        #     act = np.array([curr_steer, 0, -curr_accel])
        #
        # act = np.clip(act, -1, 1)

        return self.env.step(act)

