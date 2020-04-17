from rlpyt.agents.qpg.r2dpg_agent import R2dpgAgent
from rlpyt.models.dqn.deepdrive_r2d1_model import DeepdriveR2d1Model
from rlpyt.models.qpg.deepdrive_r2dpg_model import DeepdriveR2dpgQofMuMlpModel, DeepdriveR2dpgMuMlpModel
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.base import AgentStep

import torch

AgentInfo = namedarraytuple("AgentInfo", ["q", "prev_rnn_state"])


class DeepDriveR2dpgAgent(R2dpgAgent):

    def __init__(self, ModelCls=DeepdriveR2dpgMuMlpModel, QModelCls=DeepdriveR2dpgQofMuMlpModel,  **kwargs):
        super().__init__(ModelCls=ModelCls, QModelCls=QModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)
