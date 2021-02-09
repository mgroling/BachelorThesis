from functools import partial

import tensorflow as tf
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt

from functions import getAngle, testModel_

from stable_baselines import logger
from stable_baselines.common import (
    tf_util,
    OffPolicyRLModel,
    SetVerbosity,
    TensorboardWriter,
)
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.build_graph import build_train
from stable_baselines.deepq.policies import DQNPolicy

from stable_baselines import DQN

from skimage.measure import block_reduce

import os
import sys

sys.path.append("Fish")
from rolloutEv import testExpert


class SQIL_DQN(DQN):
    def _initializeExpertBuffer(self, obs, act):
        done = np.array([[False] for i in range(0, len(obs) - 1)])
        done[-1] = True

        self.expert_buffer.extend(obs[:-1], act, np.ones(len(obs) - 1), obs[1:], done)

    def initializeExpertBuffer(self, ar_list_obs, ar_list_act):
        """
        initizalize Expert Buffer
        """
        self.expert_buffer = ReplayBuffer(sum([len(elem) for elem in ar_list_act]))

        for i in range(0, len(ar_list_act)):
            self._initializeExpertBuffer(ar_list_obs[i], ar_list_act[i])

    def learn(
        self,
        total_timesteps,
        rollout_params,
        callback=None,
        log_interval=100,
        tb_log_name="DQN",
        reset_num_timesteps=True,
        replay_wrapper=None,
        train_graph=True,
        train_plots=None,
        train_plots_path=None
    ):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        self.sample_preds = self.expert_buffer.sample(100)
        self.sample_q = []
        self.rollout_values = [[] for i in range(len(rollout_params["perc"]))]

        with SetVerbosity(self.verbose), TensorboardWriter(
            self.graph, self.tensorboard_log, tb_log_name, new_tb_log
        ) as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    self.buffer_size, alpha=self.prioritized_replay_alpha
                )
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(
                    prioritized_replay_beta_iters,
                    initial_p=self.prioritized_replay_beta0,
                    final_p=1.0,
                )
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert (
                    not self.prioritized_replay
                ), "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                initial_p=self.exploration_initial_eps,
                final_p=self.exploration_final_eps,
            )

            rewardPer1000T = []
            episode_rewards = [0.0]
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            reset = True
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for _ in range(total_timesteps):
                if _ % 1000 == 0 and _ != 0 or _ == total_timesteps - 1:
                    print("timestep", _, "finished")
                    k = 0
                    for l in rollout_params["perc"]:
                        self.rollout_values[k].append(
                            testExpert(
                                paths=[
                                    "Fish/Guppy/validationData/" + elem
                                    for elem in os.listdir("Fish/Guppy/validationData")
                                ],
                                model=self,
                                env=self.env,
                                exp_turn_fraction=rollout_params["exp_turn_fraction"],
                                exp_speed=rollout_params["exp_min_dist"],
                                perc=l,
                                deterministic=True,
                            )
                        )
                        k += 1
                    obs = self.env.reset()
                if not train_plots is None:
                    if _ % train_plots == 0:
                        testModel_(self, train_plots_path, rollout_params, _)

                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.0
                else:
                    update_eps = 0.0
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(
                        1.0
                        - self.exploration.value(self.num_timesteps)
                        + self.exploration.value(self.num_timesteps)
                        / float(self.env.action_space.n)
                    )
                    kwargs["reset"] = reset
                    kwargs[
                        "update_param_noise_threshold"
                    ] = update_param_noise_threshold
                    kwargs["update_param_noise_scale"] = True
                with self.sess.as_default():
                    action = self.act(
                        np.array(obs)[None], update_eps=update_eps, **kwargs
                    )[0]
                env_action = action

                # check if next state (after action) would be outside of the field (CLIPPING)
                env_state = self.env.get_state()
                coords = self.env.action(env_action)
                changed = False
                if coords[0] < -0.49:
                    coords[0] = -0.47
                    changed = True
                elif coords[0] > 0.49:
                    coords[0] = 0.47
                    changed = True

                if coords[1] < -0.49:
                    coords[1] = -0.47
                    changed = True
                elif coords[1] > 0.49:
                    coords[1] = 0.47
                    changed = True

                if changed:
                    dist = np.sqrt(
                        np.power(coords[0] - env_state[0][0], 2)
                        + np.power(coords[1] - env_state[0][1], 2)
                    )
                    temp_x, temp_y = (
                        env_state[0][0] + 0.1 * np.cos(env_state[0][2]),
                        env_state[0][1] + 0.1 * np.sin(env_state[0][2]),
                    )
                    look_vec = temp_x - env_state[0][0], temp_y - env_state[0][1]
                    move_vec = coords[0] - env_state[0][0], coords[1] - env_state[0][1]
                    turn = getAngle(look_vec, move_vec, mode="radians")
                    if turn > np.pi:
                        turn = turn - 2 * np.pi
                    # convert to DQN output
                    dist_turn = np.abs(self.env.turn_rate_bins - turn)
                    dist_dist = np.abs(self.env.speed_bins - dist)

                    bin_turn = np.argmin(dist_turn, axis=0)
                    bin_dist = np.argmin(dist_dist, axis=0)

                    env_action = action = bin_turn * len(self.env.speed_bins) + bin_dist

                # for training plot/sampling
                if train_graph:
                    tempObservation = np.array(obs)
                    tempVectorized_env = self._is_vectorized_observation(
                        tempObservation, self.observation_space
                    )

                    tempObservation = tempObservation.reshape(
                        (-1,) + self.observation_space.shape
                    )
                    with self.sess.as_default():
                        tempAction, tempQ_values, tempy = self.step_model.step(
                            tempObservation
                        )
                        rewardPer1000T.append(tempQ_values[0, tempAction])

                    if _ % 1000 == 0:
                        self.sample_q.append([])
                        for sample_nr in range(0, len(self.sample_preds)):
                            (
                                obs_t_sample,
                                actions_sample,
                                rewards_sample,
                                obses_tp1_sample,
                                dones_sample,
                            ) = self.sample_preds

                            obs_t_sample = np.array(obs_t_sample[sample_nr])
                            tempVectorized_env = self._is_vectorized_observation(
                                obs_t_sample, self.observation_space
                            )

                            obs_t_sample = obs_t_sample.reshape(
                                (-1,) + self.observation_space.shape
                            )
                            with self.sess.as_default():
                                tempAction, tempQ_values, tempy = self.step_model.step(
                                    obs_t_sample
                                )
                                self.sample_q[-1].append(
                                    tempQ_values[0, actions_sample[sample_nr]]
                                )

                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                self.num_timesteps += 1

                # Stop training if return value is False
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                # Store transition in the replay buffer, but change reward to 0 (use it for plot later though)
                self.replay_buffer.add(obs_, action, 0, new_obs_, float(done))
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(
                        self.episode_reward, ep_rew, ep_done, writer, self.num_timesteps
                    )

                episode_rewards[-1] += reward_
                if done:
                    maybe_is_success = info.get("is_success")
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if (
                    can_sample
                    and self.num_timesteps > self.learning_starts
                    and self.num_timesteps % self.train_freq == 0
                ):

                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert (
                            self.beta_schedule is not None
                        ), "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(
                            self.batch_size,
                            beta=self.beta_schedule.value(self.num_timesteps),
                            env=self._vec_normalize_env,
                        )
                        (
                            obses_t,
                            actions,
                            rewards,
                            obses_tp1,
                            dones,
                            weights,
                            batch_idxes,
                        ) = experience
                    else:
                        (
                            obses_t,
                            actions,
                            rewards,
                            obses_tp1,
                            dones,
                        ) = self.replay_buffer.sample(
                            self.batch_size, env=self._vec_normalize_env
                        )
                        # also sample from expert buffer
                        (
                            obses_t_exp,
                            actions_exp,
                            rewards_exp,
                            obses_tp1_exp,
                            dones_exp,
                        ) = self.expert_buffer.sample(
                            self.batch_size, env=self._vec_normalize_env
                        )
                        weights, batch_idxes = np.ones_like(rewards), None
                        weights_exp, batch_idxes_exp = np.ones_like(rewards_exp), None
                    # pytype:enable=bad-unpacking

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(
                                trace_level=tf.RunOptions.FULL_TRACE
                            )
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(
                                np.append(obses_t, obses_t_exp, axis=0),
                                np.append(actions, actions_exp.flatten(), axis=0),
                                np.append(rewards, rewards_exp.flatten(), axis=0),
                                np.append(obses_tp1, obses_tp1_exp, axis=0),
                                np.append(obses_tp1, obses_tp1_exp, axis=0),
                                np.append(dones.flatten(), dones_exp.flatten(), axis=0),
                                np.append(weights, weights_exp),
                                sess=self.sess,
                                options=run_options,
                                run_metadata=run_metadata,
                            )
                            writer.add_run_metadata(
                                run_metadata, "step%d" % self.num_timesteps
                            )
                        else:
                            summary, td_errors = self._train_step(
                                np.append(obses_t, obses_t_exp, axis=0),
                                np.append(actions, actions_exp.flatten(), axis=0),
                                np.append(rewards, rewards_exp.flatten(), axis=0),
                                np.append(obses_tp1, obses_tp1_exp, axis=0),
                                np.append(obses_tp1, obses_tp1_exp, axis=0),
                                np.append(dones.flatten(), dones_exp.flatten(), axis=0),
                                np.append(weights, weights_exp),
                                sess=self.sess,
                                options=run_options,
                                run_metadata=run_metadata,
                            )
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(
                            np.append(obses_t, obses_t_exp, axis=0),
                            np.append(actions, actions_exp.flatten(), axis=0),
                            np.append(rewards, rewards_exp.flatten(), axis=0),
                            np.append(obses_tp1, obses_tp1_exp, axis=0),
                            np.append(obses_tp1, obses_tp1_exp, axis=0),
                            np.append(dones.flatten(), dones_exp.flatten(), axis=0),
                            np.append(weights, weights_exp),
                            sess=self.sess,
                        )

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                        self.replay_buffer.update_priorities(
                            batch_idxes, new_priorities
                        )

                    callback.on_rollout_start()

                if (
                    can_sample
                    and self.num_timesteps > self.learning_starts
                    and self.num_timesteps % self.target_network_update_freq == 0
                ):
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(
                        float(np.mean(episode_rewards[-101:-1])), 1
                    )

                num_episodes = len(episode_rewards)
                if (
                    self.verbose >= 1
                    and done
                    and log_interval is not None
                    and len(episode_rewards) % log_interval == 0
                ):
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular(
                        "% time spent exploring",
                        int(100 * self.exploration.value(self.num_timesteps)),
                    )
                    logger.dump_tabular()

        if train_graph:
            every_nth = 1000
            rewardPer1000T = np.mean(
                np.array(
                    rewardPer1000T[: (len(rewardPer1000T) // every_nth) * every_nth]
                ).reshape(-1, every_nth),
                axis=1,
            )
            x = np.arange(0, total_timesteps / every_nth)
            plt.plot(x, rewardPer1000T)
            plt.show()

            q_mean = np.mean(np.array(self.sample_q), axis=1)
            plt.plot(x, q_mean)
            plt.show()

        callback.on_training_end()
        return self