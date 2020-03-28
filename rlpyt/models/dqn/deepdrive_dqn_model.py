
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
            dueling=False
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        input_shape = observation_shape[0]
        self.base_net = torch.nn.Linear(input_shape, fc_sizes)
        # self.base_net = MlpModel(input_shape, 256, fc_sizes)

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
        x = torch.relu(self.base_net(observation))
        q = self.head(x)

        return q
