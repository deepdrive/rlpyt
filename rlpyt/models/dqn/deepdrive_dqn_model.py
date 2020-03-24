
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class DeepDriveDqnModel(torch.nn.Module):
    """
    Mlp network for DQN.
    """

    def __init__(
            self,
            observation_shape,
            output_size,
            fc_sizes=256,
            dueling=False,
            use_maxpool=False
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        in_shape = observation_shape
        # self.conv = Conv2dModel(
        #     in_channels=c,
        #     channels=channels or [32, 64, 64],
        #     kernel_sizes=kernel_sizes or [8, 4, 3],
        #     strides=strides or [4, 2, 1],
        #     paddings=paddings or [0, 1, 1],
        #     use_maxpool=use_maxpool,
        # )
        # conv_out_size = self.conv.conv_out_size(h, w)
        input_shape = observation_shape[0]
        self.fc1 = torch.nn.Linear(input_shape, fc_sizes)
        if dueling:
            self.head = DuelingHeadModel(fc_sizes, fc_sizes, output_size)
        else:
            self.head = MlpModel(fc_sizes, fc_sizes, output_size)

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
        x = torch.relu(self.fc1(observation))
        q = self.head(x)

        return q
