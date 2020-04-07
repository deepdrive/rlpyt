
import copy
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplayBuffer

configs = dict()

config = dict(
    agent=dict(),
    model=dict(dueling=True),
    algo=dict(
        discount=0.997,
        batch_T=80,
        batch_B=32,  # In the paper, 64.
        warmup_T=40,
        store_rnn_state_interval=40,
        replay_ratio=1,  # In the paper, more like 0.8.
        replay_size=int(5e5),
        learning_rate=8e-5,
        clip_grad_norm=80.,  # 80 (Steven.)
        min_steps_learn=int(1e4),
        eps_steps=int(1e6),
        target_update_interval=2500, #2500
        double_dqn=True,
        frame_state_space=False,
        prioritized_replay=True,
        input_priorities=False, ## True
        n_step_return=5, #5 #in the prioritization formula, r2d1 uses n-step return td-error -> I think we have to use n_step if we want to use prioritized replay
        pri_alpha=0.6,  # Fixed on 20190813
        pri_beta_init=0.9,  # I think had these backwards before.
        pri_beta_final=0.9,
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
        contain_prev_actions_in_obs=False,
        dummy_accel_agent_indices=[1] #for opponent
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
        contain_prev_actions_in_obs=False,
        dummy_accel_agent_indices=[1]
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
