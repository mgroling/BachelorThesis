import sys
import os
import time
import numpy as np

from wrappers import DiscreteActionWrapper, RayCastingWrapper
from convertData import getAll

from stable_baselines.deepq.policies import FeedForwardPolicy

sys.path.append("SQIL_DQN")
from SQIL_DQN_manager import SQIL_DQN_MANAGER

sys.path.append("Fish")
from functions import TestEnv


def main():
    layers, layer_norm = [[256], [256]], [False, False]
    gamma, learning_rate = [0.7, 0.7], [1e-5, 1e-5]
    buffer_size, exploration_fraction, batch_size = [1e5, 1e5], [0.3, 0.3], [32, 32]
    total_timesteps, sequential_timesteps = 25000, 1000

    env = TestEnv(steps_per_robot_action=5)
    env = RayCastingWrapper(env, degrees=360, num_bins=36)
    env = DiscreteActionWrapper(
        env,
        num_bins_turn_rate=20,
        num_bins_speed=10,
        max_turn=np.pi / 2,
        min_speed=0.00,
        max_speed=0.03,
    )

    class CustomDQNPolicy0(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomDQNPolicy0, self).__init__(
                *args,
                **kwargs,
                layers=layers[0],
                layer_norm=layer_norm[0],
                feature_extraction="mlp"
            )

    class CustomDQNPolicy1(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomDQNPolicy1, self).__init__(
                *args,
                **kwargs,
                layers=layers[1],
                layer_norm=layer_norm[1],
                feature_extraction="mlp"
            )

    model = SQIL_DQN_MANAGER(
        policy=[CustomDQNPolicy0, CustomDQNPolicy1],
        env=env,
        gamma=gamma,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        batch_size=batch_size,
    )

    # obs, act = getAll(
    #     ["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")],
    #     0,
    #     0,
    #     env,
    #     False,
    # )
    # model.initializeExpertBuffer(obs, act)

    # # TODO: Rollout, Plots while training
    # model.learn(
    #     total_timesteps=total_timesteps, sequential_timesteps=sequential_timesteps
    # )

    # TODO: save parameters
    # model.save("Fish/Guppy/models/duoDQNtest")

    model.load("Fish/Guppy/models/duoDQNtest")

    obs = env.reset()
    for i in range(100):
        action, _ = model.predict(obs)

        obs = env.step(action)[0]

        time.sleep(0.2)

        env.render()  # mode = "video"
    env.close()


if __name__ == "__main__":
    main()
