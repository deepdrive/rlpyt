
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.deepdrive_r2d1_model import DeepdriveR2d1Model
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

import torch


class DeepDriveR2d1Agent(R2d1Agent):

    def __init__(self, ModelCls=DeepdriveR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

    # @torch.no_grad()
    # def eval_step(self, observation, prev_action, prev_reward):
    #     """Computes Q-values for states/observations and selects actions by
    #     epsilon-greedy. (no grad)"""
    #     # prev_action = self.distribution.to_onehot(prev_action)
    #     model_inputs = buffer_to((observation, prev_action, prev_reward),
    #                              device=self.device)
    #     q = self.model(*model_inputs)
    #     q = q.cpu()
    #     action = torch.argmax(q)
    #     return action

    @torch.no_grad()
    def eval_step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy (no grad).  Advances RNN state."""
        prev_action = self.distribution.to_onehot(prev_action)
        prev_reward = prev_reward.float()
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)  # Model handles None.
        q = q.cpu()
        action = torch.argmax(q)
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
        # agent_info = AgentInfo(q=q, prev_rnn_state=prev_rnn_state)
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return action
        # return AgentStep(action=action, agent_info=agent_info)
