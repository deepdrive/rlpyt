
import copy
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer

configs = dict()

config = dict(
    agent=dict(),
    model=dict(),
    algo=dict(
        discount=0.997,
        batch_T=80,
        batch_B=32,  # In the paper, 64.
        warmup_T=40,
        store_rnn_state_interval=40,
        replay_ratio=8,  # In the paper, more like 0.8.
        replay_size=int(1e5),
        learning_rate=5e-4,
        clip_grad_norm=10.,  # 80 (Steven.)
        min_steps_learn=int(5e3),
        eps_steps=int(1e4),
        target_update_interval=50,
        double_dqn=True,
        frame_state_space=False,
        prioritized_replay=False,
        input_priorities=False, ## True
        n_step_return=1,
        pri_alpha=0.9,  # Fixed on 20190813
        pri_beta_init=0.6,  # I think had these backwards before.
        pri_beta_final=0.6,
        replay_buffer_class=None, #UniformSequenceReplayBuffer,
        input_priority_shift=1,  # Added 20190826 (used to default to 1)
    ),
    optim=dict(),
    env = dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=3.3e-6,
        gforce_penalty_coeff=0.006,
        lane_penalty_coeff=0.1, #0.02,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        end_on_harmful_gs=False,
        incent_win=True,
        incent_yield_to_oncoming_traffic=True,
        constrain_controls=False,
        physics_steps_per_observation=12,
        dummy_accel_agent_indices=[1],
    ),
    eval_env=dict(
        id='deepdrive-2d-intersection-w-gs-allow-decel-v0',
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=3.3e-6,
        gforce_penalty_coeff=0.006,
        lane_penalty_coeff=0.02,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        end_on_harmful_gs=False,
        incent_win=True,
        incent_yield_to_oncoming_traffic=True,
        constrain_controls=False,
        physics_steps_per_observation=12,
        dummy_accel_agent_indices=[1],
    ),
    runner=dict(
        n_steps=10e6,
        log_interval_steps=1e1,
    ),
    sampler=dict(
        batch_T=30,  # Match the algo / replay_ratio.
        batch_B=32,
        max_decorrelation_steps=1000,
        eval_n_envs=4,
        eval_max_steps=int(51e3),
        eval_max_trajectories=100,
    ),
)

configs["r2d1"] = config


config = copy.deepcopy(configs["r2d1"])
config["algo"]["replay_size"] = int(4e6)  # Even bigger is better (Steven).
config["algo"]["batch_B"] = 64  # Not sure will fit.
config["algo"]["replay_ratio"] = 1
# config["algo"]["eps_final"] = 0.1  # Now in agent.
# config["algo"]["eps_final_min"] = 0.0005
config["agent"]["eps_final"] = 0.1  # (Steven: 0.4 - 0.4 ** 8 =0.00065)
config["agent"]["eps_final_min"] = 0.0005  # (Steven: approx log space but doesn't matter)
config["runner"]["n_steps"] = 20e9
config["runner"]["log_interval_steps"] = 10e6
config["sampler"]["batch_T"] = 40  # = warmup_T = store_rnn_interval; new traj at boundary.
config["sampler"]["batch_B"] = 192  # to make one update per sample batch.
config["sampler"]["eval_n_envs"] = 6  # 6 cpus, 6 * 32 = 192, for pabti.
config["sampler"]["eval_max_steps"] = int(28e3 * 6)
config["env"]["episodic_lives"] = False  # Good effects some games (Steven).
configs["r2d1_long"] = config

config = copy.deepcopy(configs["r2d1_long"])
config["runner"]["n_steps"] = 1e6
config["runner"]["log_interval_steps"] = 1e6
config["algo"]["min_steps_learn"] = 5e4
config["sampler"]["eval_max_trajectories"] = 2
configs["r2d1_profile"] = config

config = copy.deepcopy(configs["r2d1_profile"])
config["algo"]["batch_B"] = 32
config["algo"]["replay_ratio"] = 0.5
configs["r2d1_prof_halftrain"] = config

config = copy.deepcopy(configs["r2d1_profile"])
config["algo"]["batch_B"] = 16
config["algo"]["replay_ratio"] = 0.25
configs["r2d1_prof_quartertrain"] = config

config = copy.deepcopy(configs["r2d1_long"])
config["algo"]["replay_ratio"] = 4
config["sampler"]["eval_n_envs"] = 12
config["sampler"]["eval_max_steps"] = int(28e3 * 12)
configs["r2d1_long_4tr"] = config


config = copy.deepcopy(configs["r2d1_long"])
config["sampler"]["batch_B"] = 256
config["sampler"]["batch_T"] = 40
config["algo"]["replay_size"] = int(4e6)
config["algo"]["batch_B"] = 64  # But scales with # GPUs!
config["sampler"]["eval_n_envs"] = 20
config["sampler"]["eval_max_steps"] = int(28e3 * 20)
configs["async_gpu"] = config


config = copy.deepcopy(configs["r2d1_long"])
config["sampler"]["batch_B"] = 8
config["sampler"]["batch_T"] = 5
config["sampler"]["max_decorrelation_steps"] = 20
config["env"]["episodic_lives"] = True  # To test more resets.
config["algo"]["replay_size"] = int(1e5)
config["algo"]["batch_B"] = 16
config["algo"]["batch_T"] = 10
config["algo"]["warmup_T"] = 5
config["algo"]["min_steps_learn"] = 1e4
config["algo"]["eps_steps"] = 1e5
config["algo"]["store_rnn_state_interval"] = 5
config["sampler"]["eval_n_envs"] = 8
config["sampler"]["eval_max_steps"] = int(8 * 1.5e3)
config["sampler"]["eval_max_trajectories"] = 8
config["runner"]["log_interval_steps"] = 1e5

configs["r2d1_test"] = config


config = copy.deepcopy(configs["async_gpu"])
config["sampler"]["batch_B"] = 264  # For using full maching with 2 gpu sampler, 1 gpu opt.
config["sampler"]["eval_n_envs"] = 44
config["sampler"]["eval_max_steps"] = int(44 * 28e3)  # At least one full length.
config["sampler"]["eval_max_trajectories"] = 120  # Try not to bias towards shorter ones.
configs["async_alt_pabti"] = config

config = copy.deepcopy(configs["async_gpu"])
config["sampler"]["batch_B"] = 312  # For using full maching with 3 gpu sampler, 1 gpu opt.
config["sampler"]["eval_n_envs"] = 78
config["sampler"]["eval_max_steps"] = int(78 * 28e3)  # At least one full length.
config["sampler"]["eval_max_trajectories"] = 210  # Try not to bias towards shorter ones.
configs["async_alt_dgx"] = config

config = copy.deepcopy(configs["async_gpu"])
config["sampler"]["batch_B"] = 252  # For using full maching with 2 gpu sampler, 1 gpu opt.
config["sampler"]["eval_n_envs"] = 36
config["sampler"]["eval_max_steps"] = int(36 * 28e3)  # At least one full length.
config["sampler"]["eval_max_trajectories"] = 100  # Try not to bias towards shorter ones.
configs["async_alt_got"] = config

config = copy.deepcopy(configs["async_gpu"])
config["sampler"]["batch_B"] = 312  # For using full maching with 3 gpu sampler, 1 gpu opt.
config["sampler"]["eval_n_envs"] = 39
config["sampler"]["eval_max_steps"] = int(39 * 28e3)  # At least one full length.
config["sampler"]["eval_max_trajectories"] = 210  # Try not to bias towards shorter ones.
configs["async_gpu_dgx"] = config
