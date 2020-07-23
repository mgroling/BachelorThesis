# BachelorThesis



## Introduction

### Learn models of fish behaviour by imitation learning via reinforcement learning

The goal of this bachelor's thesis is to evaluate and potentially adapt a recently presented approach for imitation learning (IL) [1] for modelling fish behaviour from existing fish motion data. To this end, the proposed method should be implemented (existing code could serve as a reference) where existing implementations of reinforcement learning algorithms can be used. A potential problem that could be encountered is that the IL method assumes a fully observable environment which is probably not the case for our problem. To overcome this issue, adaptations of the IL method should be developed, implemented, and evaluated.

[1] Reddy, Dragan, Levine, SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards, ICLR 2020

## SQIL

Good paper explaining SQIL: https://openreview.net/pdf?id=S1xKd24twB

For implementing SQIL I used an Adaption of a DQN from [stable-baselines](https://stable-baselines.readthedocs.io/en/master/) and changed it according to my needs.

## Dependencies

* Python 3.6.8
* numpy
* pandas
* tensorflow 1.15.0
* stable-baselines 2.10.0
* [my custom cartpole environment](https://github.com/marc131183/gym-Cartpole)
* [my SQIL_DQN implementation](https://github.com/marc131183/BachelorThesis/tree/master/SQIL_DQN)
* gym
* matplotlib.pyplot
* os
* sys
* math
* time
* functools
* skimage.measure.block_reduce

## Cartpole

### Introduction

The ultimate goal of my BachelorThesis was to create a SQIL model to learn fish behaviour, however first I wanted to implement the model and get used to it. For this purpose I first wanted to use Cartpole as an environment to learn a SQIL model and to verify that my implementation of SQIl works correctly. However the standard cartpole scenario (from gym) seemed a little easy for an AI, (even with a very low amount of data a model would be pretty perfect) hence I made it a little bit harder. I did not let the pole start upwards but instead downwards, so the agent would have to get it up first and then balance it. However this cannot be done in one go, first one has to spin it up in direction and then the other, in order to get the pole upwards. Implementation of this gym can be found [here](https://github.com/marc131183/gym-Cartpole).

### Trajectories

Trajectory files can be found in the data folder, they are either left_right or right_left, which means how the pole was swung up. (left_right means that it was first swung up left and then right and vice versa) All trajectories were created with the spawn of the cart in the center.

There are 4 trajectories for left_right/right_left each. However I believe that right_left trajectories are a little bit worse, (I had a harder time at creating these, because for some reason it felt more natural to me to do left_right rather than right_left) which means that models trained on right_left could be worse because of that reason.

### Model Naming

#### Model Naming of SQIL models

structure: w1_w2_w3_w4_w5_w6

w1 can only be SQIL\
w2 are the trajectories used (e.g. lr0 is trajectory_left_right_0, see in data folder)\
w3 is the start_criteria when using cutTrajectory, it is translated as follows: number at the beginning is the column, next: s = "smaller" and b = "bigger", finally: -2 is translated to /2 and rest is normal numbers/methods\
w4 is the number of timesteps that were cut off from the trajectory at the end\
w5 are the timesteps used for exploration (in learn)\
w6 determines if starting position of the cartpole is "random" or "center"

example: SQIL_lrALL_2sPi-2_40_100k_random

SQIL stands for that SQIL was used\
lrAll means that all left_right trajectories were used\
2sPi-2 means that start_criteria looked like this: (2, "smaller", Pi/2)\
40 means that cut_last = 40\
100k means that there were 100.000 exploration timesteps\
random means that position of cartpole is random at the beginning (min = -2.3, max = 2.3)

#### Model Naming of DQN/DQN_pretrained models

DQN_pretrained uses the same kind of naming as SQIL models.

DQN only has timesteps used for exploration and starting position ("center" or "random")
