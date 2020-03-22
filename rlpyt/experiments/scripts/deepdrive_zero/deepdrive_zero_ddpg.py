"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from deepdrive_zero.envs.env import Deepdrive2DEnv

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.async_.serial_sampler import AsyncSerialSampler
# from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.envs.gym import GymEnvWrapper



def make_env(*args, **kwargs):
    env = Deepdrive2DEnv()
    env.configure_env(kwargs)
    # env.render()
    return GymEnvWrapper(env)


def build_and_train(run_ID=0, cuda_idx=None):
    env_config = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        # id = 'deepdrive-2d-v0',
        is_intersection_map=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=0,
        gforce_penalty_coeff=0,
        end_on_harmful_gs=False,
        incent_win=True,
        constrain_controls=False,
    )

    sampler = SerialSampler(
        EnvCls=make_env,
        env_kwargs=env_config,
        eval_env_kwargs=env_config,
        batch_T=4,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )

    algo = DDPG(bootstrap_timelimit=False)  # SAC()  # Run with defaults.
    agent = DdpgAgent()  # SacAgent()

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e7,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),

    )

    config = dict(env_id=env_config['id'])
    name = "ddpg_" + env_config['id']
    log_dir = "dd2d"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )

