from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.deepdrive_dqn_model import DeepDriveDqnModel


class DeepDriveDqnAgent(DqnAgent):
    def __init__(self, ModelCls=DeepDriveDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    # def make_env_to_model_kwargs(self, env_spaces):
    #     assert len(env_spaces.action.shape) == 1
    #     return dict(
    #         observation_shape=env_spaces.observation.shape,
    #         action_size=env_spaces.action.shape[0],
    #     )

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)