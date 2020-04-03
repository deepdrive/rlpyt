from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.deepdrive_dqn_model import DeepDriveDqnModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.agents.base import AgentStep
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.dqn.deepdrive.mixin import DeepDriveMixin
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin

import torch


AgentInfo = namedarraytuple("AgentInfo", "q")


class DeepDriveDqnAgent(DqnAgent):
    def __init__(self, ModelCls=DeepDriveDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

    @torch.no_grad()
    def eval_step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy. (no grad)"""
        # prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q = self.model(*model_inputs)
        q = q.cpu()
        action = torch.argmax(q)
        return action