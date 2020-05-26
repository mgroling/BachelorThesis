import gym
import time
import numpy as np

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.gail import ExpertDataset
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")

if __name__ == "__main__":
    #load expert dataset
    expert_data = np.load("I:\Code\BachelorThesis\cartpole\data\expert_cartpole.npz")
    expert_array = []
    for i in range(0, len(expert_data.files)):
        expert_array.append(expert_data[expert_data.files[i]][40:60])
    len_expert_set = len(expert_array[0])
    print(len_expert_set)

    env = gym.make("CartPole-v1")

    #train network on initial dataset
    dataset = ExpertDataset(expert_path = "I:\Code\BachelorThesis\cartpole\data\expert_cartpole.npz", traj_limitation=1, batch_size=128)
    model = DQN(CustomDQNPolicy, env, verbose=1, double_q = False)
    model.pretrain(dataset, n_epochs=1000)

    #see how good it is right now
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

    #train the network by exploring
    for i in range(0, 5):
        #let agent explore
        generate_expert_traj(model, 'I:\Code\BachelorThesis\cartpole\data\explore_cartpole', n_episodes = int(len_expert_set/10))

        #save and load, cause of bug with pretrain
        model.save("I:\Code\BachelorThesis\cartpole\data\deepq_cartpole")
        del model
        model = DQN.load("I:\Code\BachelorThesis\cartpole\data\deepq_cartpole")
        model.set_env(env)
        
        #load explore dataset (same amount of rows as expert data set) and change all rewards to 0
        explore_data = np.load("I:\Code\BachelorThesis\cartpole\data\explore_cartpole.npz")
        explore_array = []
        for j in range(0, len(explore_data.files)):
            explore_array.append(explore_data[explore_data.files[j]][:len_expert_set])
        explore_array[2] = [0 for i in range(0, len(explore_array[0]))]

        #Now create a single npz file with both expert and explore data
        np.savez_compressed("I:\Code\BachelorThesis\cartpole\data\expert_explore_cartpole.npz", actions = np.array(list(expert_array[0]) + list(explore_array[0])), obs = np.array(list(expert_array[1]) + list(explore_array[1])), rewards = np.array(list(expert_array[2]) + list(explore_array[2])), episode_returns = np.array(list(expert_array[3]) + list(explore_array[3])), episode_starts = np.array(list(expert_array[4]) + list(explore_array[4])))

        #Now train the network on this expert/explore data
        temp = np.load("I:\Code\BachelorThesis\cartpole\data\expert_explore_cartpole.npz")
        train_dataset = ExpertDataset(expert_path = "I:\Code\BachelorThesis\cartpole\data\expert_explore_cartpole.npz", traj_limitation=1, batch_size=128)
        model.pretrain(train_dataset, n_epochs = 1000)

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
