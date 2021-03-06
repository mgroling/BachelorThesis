from SQIL_DQN_worker import SQIL_DQN_worker
from stable_baselines import DQN
from stable_baselines.common.buffers import ReplayBuffer
import numpy as np
import os
import sys

from rolloutEv import testExpert

sys.path.append("Fish")
from functions import testModel_


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
            )
            for i in range(2)
        ]
        for i in range(len(self._models)):
            self._models[i].action_space = env.action_space[i]
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
        self.rollout_values = [[] for i in range(len(rollout_params["perc"]))]
        for i in range(int(total_timesteps / sequential_timesteps)):
            # Train models in sequence for sequential_timesteps each
            # First model predicts turn values and second model predicts speed values
            self._models[0].learn(
                total_timesteps=sequential_timesteps,
                rollout_params=rollout_params,
                model_coworker=self._models[1],
                role="turn",
                train_plots=train_plots,
                train_plots_path=train_plots_path,
                rollout_timesteps=rollout_timesteps,
            )

            self._models[1].learn(
                total_timesteps=sequential_timesteps,
                rollout_params=rollout_params,
                model_coworker=self._models[0],
                role="speed",
                train_plots=train_plots,
                train_plots_path=train_plots_path,
                rollout_timesteps=rollout_timesteps,
            )

            if not rollout_params is None:
                if rollout_timesteps is None or total_timesteps - i < rollout_timesteps:
                    print("computing rollout values")
                    for k, l in enumerate(rollout_params["perc"]):
                        self.rollout_values[k].append(
                            testExpert(
                                paths=[
                                    "Fish/Guppy/validationData/" + elem
                                    for elem in os.listdir("Fish/Guppy/validationData")
                                ],
                                model=self,
                                env=self.env,
                                perc=l,
                                deterministic=rollout_deterministic,
                                convMat=True,
                            )
                        )

            if not train_plots is None and not rollout_params is None:
                if _ % train_plots == 0:
                    testModel_(self, train_plots_path, rollout_params, i)

            print("timestep", (i + 1) * sequential_timesteps, "finished")

    def predict(self, observation, deterministic=True):
        turn, _ = self._models[0].predict(observation, deterministic=deterministic)
        speed, _ = self._models[1].predict(observation, deterministic=deterministic)

        return [turn, speed], _

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