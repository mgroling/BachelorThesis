# BachelorThesis

## SQIL

Good paper explaining SQIL: https://openreview.net/pdf?id=S1xKd24twB

## Cartpole

### Trajectories

Trajectory files can be found in the data folder, they are either left_right or right_left, which means how the pole was swung up (left_right means that it was first swung up left and then right and vice versa)

There are 4 trajectories for left_right/right_left each. However I believe that right_left trajectories are a little bit worse, (I had a harder time at creating these, because for some reason it felt more natural to me to do left_right rather than right_left) which means that models trained on right_left could be worse because of that reason.

### Model Naming

#### Model Naming of SQIL models

structure: w1_w2_w3_w4_w5_w6

w1 can only be SQIL\
w2 are the trajectories used (e.g. lr0 is trajectory_left_right_0, see in data folder)\
w3 is the start_criteria when using cutTrajectory, it is translated as follows: number at the beginning is the column, next: s = "smaller" and b = "bigger", finally: -2 is translated to /2 and rest is normal numbers/methods\
w4 is the number of timesteps that were cut off from the trajectory at the end\
w5 are the timesteps used for exploration (in learn)\
w6 determines if starting position of the cartpole is "random" or "center"\

example: SQIL_lrALL_2sPi-2_40_100k_random

SQIL stands for that SQIL was used\
lrAll means that all left_right trajectories were used\
2sPi-2 means that start_criteria looked like this: (2, "smaller", Pi/2)\
40 means that cut_last = 40\
100k means that there were 100.000 exploration timesteps\
random means that position of cartpole is random at the beginning (min = -2.3, max = 2.3)\

#### Model Naming of DQN/DQN_pretrained models

DQN_pretrained uses the same kind of naming as SQIL models.

DQN only has timesteps used for exploration and starting position ("center" or "random")
