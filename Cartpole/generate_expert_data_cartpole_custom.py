#!/usr/bin/env python
import sys, gym, time
import gym_cartpole
import numpy as np
import pandas as pd
import os

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('cartpole_custom-v0' if len(sys.argv)<2 else sys.argv[1])#MountainCar-v0
env._max_episode_steps = 500
env.reset()
env.render("rgb_array")

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

data = None

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause, data
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    i = 0
    
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        #save observation, reward
        if i == 0:
            data = np.append(np.array([obser]), np.array([[a]]), axis = 1)
            i+=1
        else:
            data = np.append(data, np.append(np.array([obser]), np.array([[a]]), axis = 1), axis = 0)

        if r != 0:
            print("Timestep " + str(total_timesteps) + "reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

for i in range(1):
    window_still_open = rollout(env)
    save = input("save trajectory? (y/n): ")
    if save == "y":
        data_old = pd.read_csv("I:/Code/BachelorThesis/BachelorThesis/Cartpole/data/cartpole_custom_expert.csv", sep = ";").to_numpy()[:, 1:]
        data_both = np.append(data_old, data, axis = 0)
        df_both = pd.DataFrame(data = data_both)
        df_both.to_csv("I:/Code/BachelorThesis/BachelorThesis/Cartpole/data/cartpole_custom_expert.csv", sep = ";")
    if window_still_open==False: break