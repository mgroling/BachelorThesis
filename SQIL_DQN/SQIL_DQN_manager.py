from SQIL_DQN_worker import SQIL_DQN_worker
from stable_baselines import DQN
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.deepq import CnnPolicy
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from rolloutEv import testExpert

sys.path.append("Fish")
from functions import testModel_
from wrappers import DiscreteActionWrapper, RayCastingWrapper

from gym_guppy import (
    GuppyEnv,
    TurnSpeedRobot,
)
from _marc_guppy import MarcGuppyDuo


class SQIL_DQN_MANAGER:
    """
    DQN Manager for 2 DQNs
    """

    def __init__(
        self,
        policy,
        env,
        gamma,
        learning_rate,
        buffer_size,
        exploration_fraction,
        batch_size,
        seed=37,
    ):
        self._models = [
            SQIL_DQN_worker(
                policy[i],
                env,
                gamma=gamma[i],
                learning_rate=learning_rate[i],
                buffer_size=buffer_size[i],
                exploration_fraction=exploration_fraction[i],
                batch_size=batch_size[i],
                learning_starts=0,
                _init_setup_model=False,
                seed=seed,
            )
            for i in range(2)
        ]
        for i in range(len(self._models)):
            self._models[i].action_space = env.action_space.space[i]
            self._models[i].setup_model()
        self.env = env

    def initializeExpertBuffer(self, obs, act):
        for i in range(len(self._models)):
            self._models[i].expert_buffer = ReplayBuffer(
                sum([len(elem) for elem in obs])
            )
            for j in range(len(obs)):
                self._models[i]._initializeExpertBuffer(obs[j], act[j][:, i])

    def learn(
        self,
        total_timesteps,
        sequential_timesteps,
        rollout_params=None,
        rollout_timesteps=None,
        rollout_deterministic=True,
        train_plots=None,
        train_plots_path=None,
    ):
        self.rollout_values = {
            elem: [[] for i in range(len(rollout_params["perc"]))]
            for elem in rollout_params["mode"]
        }
        for i in range(int(total_timesteps / sequential_timesteps)):
            # Train models in sequence for sequential_timesteps each
            # First model predicts turn values and second model predicts speed values
            if rollout_params["learn_partner"] == "self" and i == 0:
                env = createSelfEnv(self, rollout_params)
                self._models[0].env = env
                self._models[1].env = env

            self._models[0].learn(
                total_timesteps=sequential_timesteps,
                model_coworker=self._models[1],
                role="turn",
                clipping_during_training=rollout_params["clipping_during_training"],
            )

            self._models[1].learn(
                total_timesteps=sequential_timesteps,
                model_coworker=self._models[0],
                role="speed",
                clipping_during_training=rollout_params["clipping_during_training"],
            )

            if not rollout_params is None:
                if (
                    rollout_timesteps is None
                    or total_timesteps // sequential_timesteps - i <= rollout_timesteps
                ):
                    print("computing rollout values")
                    for k, l in enumerate(rollout_params["perc"]):
                        for mode in rollout_params["mode"]:
                            self.rollout_values[mode][k].append(
                                testExpert(
                                    paths=[
                                        "Fish/Guppy/validationData/" + elem
                                        for elem in os.listdir(
                                            "Fish/Guppy/validationData"
                                        )
                                    ],
                                    model=self,
                                    env=self.env,
                                    perc=l,
                                    deterministic=rollout_deterministic,
                                    convMat=True,
                                    mode=mode,
                                )
                            )

            if not train_plots is None and not rollout_params is None:
                if _ % train_plots == 0:
                    testModel_(self, train_plots_path, rollout_params, i)

            print("timestep", (i + 1) * sequential_timesteps, "finished")

    def predict(self, observation, deterministic=True, tau=0.04):
        if deterministic:
            turn, _ = self._models[0].predict(observation)
            speed, _ = self._models[1].predict(observation)
        else:
            q_values_turn = self._models[0].step_model.step(np.array([observation]))[1][
                0
            ]
            q_values_speed = self._models[1].step_model.step(np.array([observation]))[
                1
            ][0]

            q_values_turn = q_values_turn - q_values_turn.min()
            q_values_speed = q_values_speed - q_values_speed.min()

            q_values_turn = q_values_turn / q_values_turn.max()
            q_values_speed = q_values_speed / q_values_speed.max()

            q_values_turn_exp = np.exp(q_values_turn / tau)
            q_values_speed_exp = np.exp(q_values_speed / tau)

            probabilities_turn = q_values_turn_exp / np.sum(q_values_turn_exp)
            probabilities_speed = q_values_speed_exp / np.sum(q_values_speed_exp)

            turn = np.random.choice(range(len(q_values_turn)), p=probabilities_turn)
            speed = np.random.choice(range(len(q_values_speed)), p=probabilities_speed)

        return [turn, speed], None

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self._models[0].save(save_path + "/model_turn")
        self._models[1].save(save_path + "/model_speed")

    @classmethod
    def load(cls, save_path):
        obj = cls.__new__(cls)
        obj._models = [None, None]
        obj._models[0] = DQN.load(save_path + "/model_turn")
        obj._models[1] = DQN.load(save_path + "/model_speed")
        return obj


def createSelfEnv(model, dic):
    class TestEnv(GuppyEnv):
        def _reset(self):
            # set frequency to 25Hz
            self._guppy_steps_per_action = 4

            pos = (
                np.random.uniform(low=-0.3, high=0.3),
                np.random.uniform(low=-0.3, high=0.3),
            )
            ori = np.random.uniform() * 2 * np.pi

            self._add_robot(
                TurnSpeedRobot(
                    world=self.world,
                    world_bounds=self.world_bounds,
                    position=pos,
                    orientation=ori,
                )
            )

            num_guppies = 1
            positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (
                0.05,
                0.05,
            )
            orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
            for p, o in zip(positions, orientations):
                self._add_guppy(
                    MarcGuppyDuo(
                        model=model,
                        dic=dic,
                        world=self.world,
                        world_bounds=self.world_bounds,
                        position=(
                            np.random.uniform(low=-0.3, high=0.3),
                            np.random.uniform(low=-0.3, high=0.3),
                        ),
                        orientation=np.random.uniform() * 2 * np.pi,
                    )
                )

    env = TestEnv(time_step_s=0.04)
    env = RayCastingWrapper(
        env,
        degrees=dic["degrees"],
        num_bins=dic["num_bins_rays"],
        last_act=dic["last_act"],
    )
    env = DiscreteActionWrapper(
        env,
        num_bins_turn_rate=dic["turn_bins"],
        num_bins_speed=dic["speed_bins"],
        max_turn=dic["max_turn"],
        min_speed=dic["min_speed"],
        max_speed=dic["max_speed"],
        last_act=dic["last_act"],
    )

    return env
