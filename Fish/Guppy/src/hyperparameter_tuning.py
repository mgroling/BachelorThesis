import optuna
import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from wrappers import DiscreteMatrixActionWrapper, RayCastingWrapper
from convertData import getAll
from stable_baselines.deepq.policies import FeedForwardPolicy

sys.path.append("Fish")
from functions import TestEnv

sys.path.append("SQIL_DQN")
from SQIL_DQN import SQIL_DQN

# Suggest: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial


def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layer_structure = []
    for i in range(n_layers):
        layer_structure.append(
            int(trial.suggest_loguniform("n_units_l" + str(i), 4, 512))
        )
    layer_norm = trial.suggest_categorical("layer_norm", [True, False])

    gamma = trial.suggest_uniform("gamma", 0.5, 0.999)
    lr = trial.suggest_loguniform("lr", 1e-5, 0.1)
    n_batch = trial.suggest_int("n_batch", 1, 128)

    explore_fraction = trial.suggest_uniform("explore_fraction", 0.01, 0.5)

    learn_timesteps = trial.suggest_int("learn_timesteps", 5000, 1e5, 1000)
    print("Learn timesteps", learn_timesteps)

    # Train model and evaluate it
    class CustomDQNPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomDQNPolicy, self).__init__(
                *args,
                **kwargs,
                layers=layer_structure,
                layer_norm=layer_norm,
                feature_extraction="mlp"
            )

    env = TestEnv(max_steps_per_action=200)
    env = RayCastingWrapper(env, degrees=360, num_bins=36)
    env = DiscreteMatrixActionWrapper(
        env,
        num_bins_turn_rate=20,
        num_bins_speed=10,
        max_turn=np.pi,
        min_speed=0.01,
        max_speed=0.07,
    )

    model = SQIL_DQN(
        CustomDQNPolicy,
        env,
        verbose=1,
        buffer_size=100000,
        double_q=False,
        seed=37,
        gamma=gamma,
        learning_rate=lr,
        batch_size=n_batch,
        exploration_fraction=explore_fraction,
    )

    obs, act = getAll(
        ["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")],
        np.pi / 4,
        0.01,
        env,
    )
    model.initializeExpertBuffer(obs, act)

    rollout_dic = {"perc": [0], "exp_turn_fraction": 4, "exp_min_dist": 0.01}

    model.learn(
        total_timesteps=learn_timesteps,
        rollout_params=rollout_dic,
        rollout_timesteps=5000,
        train_graph=False,
        train_plots=None,
    )

    reward = []
    for i in range(len(model.rollout_values)):
        for value in model.rollout_values[i]:
            reward.append(value[0])

    return 1 - np.mean(reward)


def saveStudy(study, path):
    with open(path, "wb") as output:
        pickle.dump(study, output, pickle.HIGHEST_PROTOCOL)


def loadStudy(path):
    with open(path, "rb") as input:
        study = pickle.load(input)
    return study


def main():
    study = optuna.create_study()

    while True:
        study.optimize(objective, n_trials=1)
        saveStudy(study, "Fish/Guppy/studies/study.pkl")

    print(study.best_params)

    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()


if __name__ == "__main__":
    main()
