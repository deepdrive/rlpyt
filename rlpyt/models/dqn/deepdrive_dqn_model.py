
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.running_mean_std import RunningMeanStdModel


class DeepDriveDqnModel(torch.nn.Module):
    """
    Mlp network for DQN.
    """

    def __init__(
            self,
            observation_shape,
            output_size,
            fc_sizes=[128, 128],
            dueling=False,
            normalize_observation=False,
            norm_obs_clip=2,
            norm_obs_var_clip=1e-6,
        ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        self._obs_ndim = len(observation_shape)
        input_shape = observation_shape[0]

        self.base_net = torch.nn.Sequential(
            torch.nn.Linear(input_shape, fc_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_sizes[0], fc_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_sizes[1], output_size),
        )
        # self.base_net.apply(self.init_weights)

        # self.base_net = MlpModel(input_shape, [fc_sizes, fc_sizes], output_size)

        # normalize obs
        # if normalize_observation:
        #     self.obs_rms = RunningMeanStdModel(observation_shape)
        #     self.norm_obs_clip = norm_obs_clip
        #     self.norm_obs_var_clip = norm_obs_var_clip
        # self.normalize_observation = normalize_observation

        # head network
        # if dueling:
        #     self.head = DuelingHeadModel(fc_sizes, fc_sizes, output_size)
        # else:
        #     self.head = MlpModel(fc_sizes, fc_sizes, output_size)

        # self.tot_mean = 0
        # self.tot_std = 0
        # self.n_data = 0

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute action Q-value estimates from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        image_shape[0], image_shape[1],...,image_shape[-1]], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        observation = observation.type(torch.float)
        lead_dim, T, B, obs_shape = infer_leading_dims(observation, self._obs_ndim)

        obs = observation.view(T * B, -1)

        # if self.normalize_observation:
        #     # obs_var = self.obs_rms.var
        #     # if self.norm_obs_var_clip is not None:
        #     #     obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
        #     # observation = torch.clamp((observation - self.obs_rms.mean) /
        #     #     obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)
        #     new_mean = obs.mean(dim=0)
        #     new_std = obs.std(dim=0)
        #
        #     self.tot_mean = (new_mean * obs.shape[1] + self.tot_mean * self.n_data) / (obs.shape[1] + self.n_data)
        #     self.tot_std = (new_std * obs.shape[1] + self.tot_std * self.n_data) / (obs.shape[1] + self.n_data)
        #
        #     obs = (obs - self.tot_mean) / (self.tot_std + 1e-8)
        #
        #     self.n_data += obs.shape[1]

        q = self.base_net(obs)
        # q = self.base_net(observation)
        # q = torch.relu(q)
        # q = self.head(x)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

    # def update_obs_rms(self, observation):
    #     if self.normalize_observation:
    #         self.obs_rms.update(observation)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.uniform_(m.bias.data)

