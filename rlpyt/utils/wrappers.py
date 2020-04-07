from rlpyt.envs.base import Env
from rlpyt.envs.gym import IntBox
import gym

class DeepDriveDiscretizeActionWrapper(gym.ActionWrapper, Env):
    """ Discretizes the action space of deepdrive_zero env.
    """
    def __init__(self, env):
        super(DeepDriveDiscretizeActionWrapper, self).__init__(env)
        discrete_steer = [-.3, -.2, -.1, 0, 0.1, .2, .3] #list(np.arange(-0.45, 0.451, 0.15)) #list(np.arange(-1, 1.01, 0.08))
        discrete_acc   = [-1, 0.5, 1]
        # discrete_brake = [-1, 0, 1]
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
                    self.action_items.append([s, a, 0])
                else:
                    self.action_items.append([s, 0, -a])

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
        return self.env.step(act)
