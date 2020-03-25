from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.deepdrive_dqn_model import DeepDriveDqnModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.agents.base import AgentStep
from rlpyt.utils.collections import namedarraytuple

import torch


AgentInfo = namedarraytuple("AgentInfo", "q")


class DeepDriveDqnAgent(DqnAgent):
    def __init__(self, ModelCls=DeepDriveDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    # def make_env_to_model_kwargs(self, env_spaces):
    #     assert len(env_spaces.action.shape) == 1
    #     return dict(
    #         observation_shape=env_spaces.observation.shape,
    #         action_size=env_spaces.action.shape[0],
    #     )

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy. (no grad)"""
        # prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q = self.model(*model_inputs)
        q = q.cpu()
        action = self.distribution.sample(q)
        agent_info = AgentInfo(q=q)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)