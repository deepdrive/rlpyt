
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


RnnState = namedarraytuple("RnnState", ["h", "c"])


class DeepdriveR2d1Model(torch.nn.Module):
    """2D convolutional neural network (for multiple video frames per
    observation) feeding into an LSTM and MLP output for Q-value outputs for
    the action set."""
    def __init__(
            self,
            observation_shape,
            output_size,
            fc_size=512,  # Between mlp and lstm.
            lstm_size=512,
            head_size=256,
            dueling=False,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self._obs_n_dim = len(observation_shape)
        self.dueling = dueling
        self.mlp = MlpModel(observation_shape, [256],
                            output_size=fc_size,
                            nonlinearity=torch.nn.Tanh  # Match spinningup
                            )
        self.lstm = torch.nn.LSTM(fc_size + output_size + 1, lstm_size)
        if dueling:
            self.head = DuelingHeadModel(lstm_size, head_size, output_size)
        else:
            self.head = MlpModel(lstm_size, head_size, output_size=output_size)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        obz = observation.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_n_dim)

        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        mlp_out = self.mlp(observation.view(T * B, -1))

        lstm_input = torch.cat([
            mlp_out.view(T, B, -1),
            prev_action.view(T, B, -1),
            prev_reward.view(T, B, 1),
            ], dim=2)

        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        q = self.head(lstm_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, next_rnn_state
