
import torch

from rlpyt.agents.base import (AgentStep, RecurrentAgentMixin,
    AlternatingRecurrentAgentMixin)
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.collections import namedarraytuple
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.qpg.mlp import MuMlpModel, QofMuMlpModel
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["q", "mu", "prev_rnn_state"])


class R2dpgAgentBase(DdpgAgent):
    """Base agent for recurrent DQN (to add recurrent mixin)."""

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]

        model_inputs = buffer_to((observation, prev_action, prev_reward,)
                                 , device=self.device)
        init_rnn_state = buffer_to(init_rnn_state, device=self.device)
        mu, _ = self.model(*model_inputs, init_rnn_state)
        q, rnn_state = self.q_model(*model_inputs, mu, init_rnn_state)
        return q.cpu(), rnn_state  # Leave rnn state on device.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy (no grad).  Advances RNN state."""
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, _ = self.model(*agent_inputs, self.prev_rnn_state)
        mu = mu.to(self.device)
        q, rnn_state = self.q_model(*agent_inputs, mu, self.prev_rnn_state)  # Model handles None.
        q = q.cpu()
        action = self.distribution.sample(DistInfo(mean=mu))
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
        agent_info = AgentInfo(q=q, mu=mu.cpu(), prev_rnn_state=prev_rnn_state)
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    def q(self, observation, prev_action, prev_reward, action, init_rnn_state):
        """Compute Q-value for input state/observation and action (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q, rnn_state = self.q_model(*model_inputs, init_rnn_state)
        return q.cpu(), rnn_state

    def q_at_mu(self, observation, prev_action, prev_reward, init_rnn_state):
        """Compute Q-value for input state/observation, through the mu_model
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, _ = self.model(*model_inputs, init_rnn_state)
        q, rnn_state = self.q_model(*model_inputs, mu, init_rnn_state)
        return q.cpu(), rnn_state

    def target_q_at_mu(self, observation, prev_action, prev_reward, init_rnn_state):
        """Compute target Q-value for input state/observation, through the
        target mu_model."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        init_rnn_state = buffer_to(init_rnn_state, device=self.device)
        target_mu, _ = self.target_model(*model_inputs, init_rnn_state)
        target_q_at_mu, rnn_state = self.target_q_model(*model_inputs, target_mu, init_rnn_state)
        return target_q_at_mu.cpu(), rnn_state


class R2dpgAgent(RecurrentAgentMixin, R2dpgAgentBase):
    """R2D1 agent."""
    pass


class R2dpgAlternatingAgent(AlternatingRecurrentAgentMixin, R2dpgAgentBase):
    pass
