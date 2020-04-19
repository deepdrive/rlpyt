
import numpy as np

from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer
from rlpyt.replays.async_ import AsyncReplayBufferMixin


class UniformSequenceReplay:
    """Replays sequences with starting state chosen uniformly randomly.
    """

    def set_batch_T(self, batch_T):
        self.batch_T = batch_T  # Can set dynamically, or input to sample_batch.

    def sample_batch(self, batch_B, batch_T=None):
        """Can dynamically input length of sequences to return, by ``batch_T``,
        else if ``None`` will use interanlly set value.  Returns batch with
        leading dimensions ``[batch_T, batch_B]``.
        """
        batch_T = self.batch_T if batch_T is None else batch_T
        T_idxs, B_idxs = self.sample_idxs(batch_B, batch_T)
        return self.extract_batch(T_idxs, B_idxs, batch_T)

    def sample_idxs(self, batch_B, batch_T):
        """Randomly choose the indexes of starting data to return using
        ``np.random.randint()``.  Disallow samples within certain proximity to
        the current cursor which hold invalid data, including accounting for
        sequence length (so every state returned in sequence will hold valid
        data).  If the RNN state is only stored periodically, only choose
        starting states with stored RNN state.
        """
        t, b, f = self.t, self.off_backward + batch_T, self.off_forward
        high = self.T - b - f if self._buffer_full else t - b - f
        T_idxs = np.random.randint(low=0, high=high, size=(batch_B,))
        T_idxs[T_idxs >= t - b] += min(t, b) + f
        if self.rnn_state_interval > 0:  # Some rnn states stored; only sample those.
            T_idxs = (T_idxs // self.rnn_state_interval) * self.rnn_state_interval
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs


class UniformSequenceReplayBuffer(UniformSequenceReplay,
        SequenceNStepReturnBuffer):
    pass


class AsyncUniformSequenceReplayBuffer(AsyncReplayBufferMixin,
        UniformSequenceReplayBuffer):
    pass


######################################## replay buffer for r2dpg to save two rnn_states for actor and critic ####
import math
import numpy as np

from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, buffer_func
from rlpyt.utils.misc import extract_sequences
from rlpyt.utils.collections import namedarraytuple
from rlpyt.replays.n_step import BaseNStepReturnBuffer

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
                                    ["all_observation", "all_action", "all_reward", "return_", "done", "done_n",
                                     "q_init_rnn_state", "mu_init_rnn_state"])

SamplesToBuffer = None

class UniformSequenceReplayBufferR2dpg(UniformSequenceReplay, BaseNStepReturnBuffer):

    """Base n-step return buffer for sequences replays.  Includes storage of
    agent's recurrent (RNN) state.

    Use of ``rnn_state_interval>1`` only periodically
    stores RNN state, to save memory.  The replay mechanism must account for the
    fact that only time-steps with saved RNN state are valid first states for replay.
    (``rnn_state_interval<1`` does not store RNN state.)
    """

    def __init__(self, example, size, B, rnn_state_interval, batch_T=None, **kwargs):
        self.rnn_state_interval = rnn_state_interval
        self.batch_T = batch_T  # Maybe required fixed depending on replay type.
        if rnn_state_interval <= 1:  # Store no rnn state or every rnn state.
            buffer_example = example
        else:
            # Store some of rnn states; remove from samples.
            field_names = [f for f in example._fields if (f != "q_prev_rnn_state" and f != "mu_prev_rnn_state")]
            global SamplesToBuffer
            SamplesToBuffer = namedarraytuple("SamplesToBuffer", field_names)
            buffer_example = SamplesToBuffer(*(v for k, v in example.items()
                                               if k != "q_prev_rnn_state" and k != "mu_prev_rnn_state"))
            size = B * rnn_state_interval * math.ceil(  # T as multiple of interval.
                math.ceil(size / B) / rnn_state_interval)
            self.samples_q_prev_rnn_state = buffer_from_example(example.q_prev_rnn_state,
                                                              (size // (B * rnn_state_interval), B),
                                                              share_memory=self.async_)
            self.samples_mu_prev_rnn_state = buffer_from_example(example.mu_prev_rnn_state,
                                                                (size // (B * rnn_state_interval), B),
                                                                share_memory=self.async_)
        super().__init__(example=buffer_example, size=size, B=B, **kwargs)
        if rnn_state_interval > 1:
            assert self.T % rnn_state_interval == 0
            self.rnn_T = self.T // rnn_state_interval

    def append_samples(self, samples):
        """Special handling for RNN state storage, and otherwise uses superclass's
        ``append_samples()``.
        """
        t, rsi = self.t, self.rnn_state_interval
        if rsi <= 1:  # All or no rnn states stored.
            return super().append_samples(samples)
        buffer_samples = SamplesToBuffer(*(v for k, v in samples.items()
                                           if k != "q_prev_rnn_state" and k != "mu_prev_rnn_state"))
        T, idxs = super().append_samples(buffer_samples)
        start, stop = math.ceil(t / rsi), ((t + T - 1) // rsi) + 1
        offset = (rsi - t) % rsi
        if stop > self.rnn_T:  # Wrap.
            rnn_idxs = np.arange(start, stop) % self.rnn_T
        else:
            rnn_idxs = slice(start, stop)
        self.samples_q_prev_rnn_state[rnn_idxs] = samples.q_prev_rnn_state[offset::rsi]
        self.samples_mu_prev_rnn_state[rnn_idxs] = samples.mu_prev_rnn_state[offset::rsi]
        return T, idxs

    def extract_batch(self, T_idxs, B_idxs, T):
        """Return full sequence of each field in `agent_inputs` (e.g. `observation`),
        including all timesteps for the main sequence and for the target sequence in
        one array; many timesteps will likely overlap, so the algorithm and make
        sub-sequences by slicing on device, for reduced memory usage.

        Enforces that input `T_idxs` align with RNN state interval.

        Uses helper function ``extract_sequences()`` to retrieve samples of
        length ``T`` starting at locations ``[T_idxs,B_idxs]``, so returned
        data batch has leading dimensions ``[T,len(B_idxs)]``."""
        s, rsi = self.samples, self.rnn_state_interval
        if rsi > 1:
            assert np.all(np.asarray(T_idxs) % rsi == 0)
            q_init_rnn_state = self.samples_q_prev_rnn_state[T_idxs // rsi, B_idxs]
            mu_init_rnn_state = self.samples_mu_prev_rnn_state[T_idxs // rsi, B_idxs]
        elif rsi == 1:
            q_init_rnn_state = self.samples.q_prev_rnn_state[T_idxs, B_idxs]
            mu_init_rnn_state = self.samples.mu_prev_rnn_state[T_idxs, B_idxs]
        else:  # rsi == 0
            init_rnn_state = None
        batch = SamplesFromReplay(
            all_observation=self.extract_observation(T_idxs, B_idxs,
                                                     T + self.n_step_return),
            all_action=buffer_func(s.action, extract_sequences, T_idxs - 1, B_idxs,
                                   T + self.n_step_return),  # Starts at prev_action.
            all_reward=extract_sequences(s.reward, T_idxs - 1, B_idxs,
                                         T + self.n_step_return),  # Only prev_reward (agent + target).
            return_=extract_sequences(self.samples_return_, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
            done_n=extract_sequences(self.samples_done_n, T_idxs, B_idxs, T),
            q_init_rnn_state=q_init_rnn_state,  # (Same state for agent and target.)
            mu_init_rnn_state=mu_init_rnn_state
        )
        # NOTE: Algo might need to make zero prev_action/prev_reward depending on done.
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs, T):
        """Generalization anticipating frame-buffer."""
        return buffer_func(self.samples.observation, extract_sequences,
                           T_idxs, B_idxs, T)

