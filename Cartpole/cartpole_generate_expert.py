import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.gail import generate_expert_traj

import time

env = gym.make('CartPole-v1')

# Custom MLP policy of two layers of size 32 each
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")

model = DQN(CustomDQNPolicy, env, verbose=1)

model.learn(total_timesteps=100000)

generate_expert_traj(model, "I:\Code\BachelorThesis\cartpole\data\expert_cartpole", n_episodes=10)

#test it
reward_sum = 0.0
obs = env.reset()
for i in range(0, 5):
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        env.render()
    print(reward_sum)
    reward_sum = 0.0
    obs = env.reset()

env.close()
