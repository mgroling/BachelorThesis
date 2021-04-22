import json
import numpy as np

from stable_baselines import DQN


class Model:
    _frequency = 25

    def __init__(self, path, deterministic=False, tau=0.03):
        self.turn_model = DQN.load(path + "model_turn")
        self.speed_model = DQN.load(path + "model_speed")
        dic = loadConfig(path + "parameters.json")

        self.tau = tau
        self.deterministic = deterministic

        self.turn_bins = np.linspace(
            -dic["max_turn"], dic["max_turn"], dic["turn_bins"]
        )
        self.speed_bins = np.linspace(
            dic["min_speed"], dic["max_speed"], dic["speed_bins"]
        )

        self.raycast_options = {
            "n_fish_bins": dic["num_bins_rays"],
            "n_wall_raycasts": dic["num_bins_rays"],
            "fov_angle_fish_bins": np.radians(dic["degreees"]),
            "fov_angle_wall_raycasts": np.radians(dic["degrees"]),
            "world_bounds": ([-50, -50], [50, 50]),
        }

    def choose_action(self, view: np.ndarray) -> Tuple[float, float]:
        obs = np.array([[view[: len(view)], view[len(view) :]]])

        if self.deterministic:
            turn, _ = self.turn_model.predict(obs[0])
            speed, _ = self.speed_model.predict(obs[0])

            turn = self.turn_bins[turn]
            speed = self.speed_bins[speed]
        else:
            q_values_turn = self.turn_model.step_model.step(obs)[1][0]
            q_values_speed = self.speed_model.step_model.step(obs)[1][0]

            q_values_turn = q_values_turn - q_values_turn.min()
            q_values_speed = q_values_speed - q_values_speed.min()

            q_values_turn = q_values_turn / q_values_turn.max()
            q_values_speed = q_values_speed / q_values_speed.max()

            q_values_turn_exp = np.exp(q_values_turn / self.tau)
            q_values_speed_exp = np.exp(q_values_speed / self.tau)

            probabilities_turn = q_values_turn_exp / np.sum(q_values_turn_exp)
            probabilities_speed = q_values_speed_exp / np.sum(q_values_speed_exp)

            turn = np.random.choice(self.turn_bins, p=probabilities_turn)
            speed = np.random.choice(self.speed_bins, p=probabilities_speed)

        return speed * self._frequency, turn * self._frequency


def saveConfig(path, dic):
    with open(path, "w+") as f:
        json.dump(dic, f)


def loadConfig(path):
    with open(path, "r") as f:
        return json.load(f)