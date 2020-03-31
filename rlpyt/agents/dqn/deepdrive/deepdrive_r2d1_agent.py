
from rlpyt.agents.dqn.r2d1_agent import R2d1Agent
from rlpyt.models.dqn.deepdrive_r2d1_model import DeepdriveR2d1Model


class DeepDriveR2d1Agent(R2d1Agent):

    def __init__(self, ModelCls=DeepdriveR2d1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
