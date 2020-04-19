
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

AgentInfo = namedarraytuple("AgentInfo", ["q", "mu", "q_prev_rnn_state", "mu_prev_rnn_state"])


class R2dpgAgentBase(DdpgAgent):
    """Base agent for recurrent DQN (to add recurrent mixin)."""

    def __call__(self, observation, prev_action, prev_reward, q_init_rnn_state, mu_init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]

        model_inputs = buffer_to((observation, prev_action, prev_reward,)
                                 , device=self.device)
        q_init_rnn_state = buffer_to(q_init_rnn_state, device=self.device)
        mu_init_rnn_state = buffer_to(mu_init_rnn_state, device=self.device)
        q_init_rnn_state = buffer_method(q_init_rnn_state, "squeeze", -2)
        mu_init_rnn_state = buffer_method(mu_init_rnn_state, "squeeze", -2)
        mu, mu_rnn_state = self.model(*model_inputs, mu_init_rnn_state)
        q, q_rnn_state = self.q_model(*model_inputs, mu, q_init_rnn_state)
        return q.cpu(), q_rnn_state, mu_rnn_state  # Leave rnn state on device.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy (no grad).  Advances RNN state."""
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, mu_rnn_state = self.model(*agent_inputs, self.mu_prev_rnn_state)
        mu = mu.to(self.device)
        q, q_rnn_state = self.q_model(*agent_inputs, mu, self.q_prev_rnn_state)  # Model handles None.
        q = q.cpu()
        action = self.distribution.sample(DistInfo(mean=mu))
        q_prev_rnn_state = self.q_prev_rnn_state or buffer_func(q_rnn_state, torch.zeros_like)
        mu_prev_rnn_state = self.mu_prev_rnn_state or buffer_func(mu_rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case, model should always leave B dimension in.)
        q_prev_rnn_state = buffer_method(q_prev_rnn_state, "transpose", 0, 1)
        q_prev_rnn_state = buffer_to(q_prev_rnn_state, device="cpu")
        mu_prev_rnn_state = buffer_method(mu_prev_rnn_state, "transpose", 0, 1)
        mu_prev_rnn_state = buffer_to(mu_prev_rnn_state, device="cpu")

        agent_info = AgentInfo(q=q, mu=mu.cpu(), q_prev_rnn_state=q_prev_rnn_state, mu_prev_rnn_state=mu_prev_rnn_state)
        self.advance_rnn_state(q_rnn_state, mu_rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    def q(self, observation, prev_action, prev_reward, action, q_init_rnn_state):
        """Compute Q-value for input state/observation and action (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q, q_rnn_state = self.q_model(*model_inputs, q_init_rnn_state)
        return q.cpu(), q_rnn_state

    def q_at_mu(self, observation, prev_action, prev_reward,
                q_init_rnn_state, mu_init_rnn_state):
        """Compute Q-value for input state/observation, through the mu_model
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, mu_rnn_state = self.model(*model_inputs, mu_init_rnn_state)
        q, q_rnn_state = self.q_model(*model_inputs, mu, q_init_rnn_state)
        return q.cpu(), q_rnn_state, mu_rnn_state

    def target_q_at_mu(self, observation, prev_action, prev_reward,
                       q_init_rnn_state, mu_init_rnn_state):
        """Compute target Q-value for input state/observation, through the
        target mu_model."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        q_init_rnn_state = buffer_to(q_init_rnn_state, device=self.device)
        mu_init_rnn_state = buffer_to(mu_init_rnn_state, device=self.device)
        q_init_rnn_state = buffer_method(q_init_rnn_state, "squeeze", -2)
        mu_init_rnn_state = buffer_method(mu_init_rnn_state, "squeeze", -2)
        target_mu, mu_rnn_state = self.target_model(*model_inputs, mu_init_rnn_state)
        target_q, q_rnn_state = self.target_q_model(*model_inputs, target_mu, q_init_rnn_state)
        return target_q.cpu(), q_rnn_state, mu_rnn_state




AgentInputsRnn = namedarraytuple("AgentInputsRnn",  # Training only.
    ["observation", "prev_action", "prev_reward", "q_init_rnn_state", "mu_init_rnn_state"])


class R2dpgRecurrentAgentMixin:
    """
    Mixin class to manage recurrent state during sampling (so the sampler
    remains agnostic).  To be used like ``class
    MyRecurrentAgent(RecurrentAgentMixin, MyAgent):``.
    """
    recurrent = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_prev_rnn_state = None
        self._mu_prev_rnn_state = None
        self._sample_q_rnn_state = None  # Store during eval.
        self._sample_mu_rnn_state = None  # Store during eval.

    def reset(self):
        """Sets the recurrent state to ``None``, which built-in PyTorch
        modules conver to zeros."""
        self._q_prev_rnn_state = None
        self._mu_prev_rnn_state = None

    def reset_one(self, idx):
        """Sets the recurrent state corresponding to one environment instance
        to zero.  Assumes rnn state is in cudnn-compatible shape: [N,B,H],
        where B corresponds to environment index."""
        if self._q_prev_rnn_state is not None:
            self._q_prev_rnn_state[:, idx] = 0  # Automatic recursion in namedarraytuple.
        if self._mu_prev_rnn_state is not None:
            self._mu_prev_rnn_state[:, idx] = 0  # Automatic recursion in namedarraytuple.

    def advance_rnn_state(self, new_q_rnn_state, new_mu_rnn_state):
        """Sets the recurrent state to the newly computed one (i.e. recurrent agents should
        call this at the end of their ``step()``). """
        self._q_prev_rnn_state = new_q_rnn_state
        self._mu_prev_rnn_state = new_mu_rnn_state

    @property
    def q_prev_rnn_state(self):
        return self._q_prev_rnn_state

    @property
    def mu_prev_rnn_state(self):
        return self._mu_prev_rnn_state

    def train_mode(self, itr):
        """If coming from sample mode, store the rnn state elsewhere and clear it."""
        if self._mode == "sample":
            self._sample_q_rnn_state = self._q_prev_rnn_state
            self._sample_mu_rnn_state = self._mu_prev_rnn_state
        self._q_prev_rnn_state = None
        self._mu_prev_rnn_state = None
        super().train_mode(itr)

    def sample_mode(self, itr):
        """If coming from non-sample modes, restore the last sample-mode rnn state."""
        if self._mode != "sample":
            self._q_prev_rnn_state = self._sample_q_rnn_state
            self._mu_prev_rnn_state = self._sample_mu_rnn_state
        super().sample_mode(itr)

    def eval_mode(self, itr):
        """If coming from sample mode, store the rnn state elsewhere and clear it."""
        if self._mode == "sample":
            self._sample_q_rnn_state = self._q_prev_rnn_state
            self._sample_mu_rnn_state = self._mu_prev_rnn_state
        self._q_prev_rnn_state = None
        self._mu_prev_rnn_state = None
        super().eval_mode(itr)



class R2dpgAgent(R2dpgRecurrentAgentMixin, R2dpgAgentBase):
    """R2DPG agent."""
    pass


class R2dpgAlternatingAgent(AlternatingRecurrentAgentMixin, R2dpgAgentBase):
    pass
