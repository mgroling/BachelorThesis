# BachelorThesis of Marc Gröling

[Link to thesis](http://www.mi.fu-berlin.de/inf/groups/ag-ki/Theses/Completed-theses/Bachelor-theses/2021/Groeling/index.html)

[Link to model](https://git.imp.fu-berlin.de/bioroboticslab/robofish/fish_models)

## Abstract

The collective behaviour of groups of animals emerges from interaction between individuals. Understanding these interindividual rules has always been a challenge, because the cognition of animals is not fully understood. Artificial neural networks in conjunction with attribution methods and others can help decipher these interindividual rules. In this thesis, an artificial neural network was trained with a recently proposed learning algorithm called Soft Q Imitation Learning (SQIL) on a dataset of two female guppies. The network  is able to outperform a simple agent that uses the action of the most similar state in defined metric and also is able to show most characteristics of fish, at least partially, when simulated.

## Used data

[Expert data](https://github.com/marc131183/BachelorThesis/tree/master/Fish/Guppy/data)

[Validation data](https://github.com/marc131183/BachelorThesis/tree/master/Fish/Guppy/validationData)

## Important files

[Train model](https://github.com/marc131183/BachelorThesis/blob/master/Fish/Guppy/src/duoDQN.py)
[Evaluate model's performance (rollout)](https://github.com/marc131183/BachelorThesis/blob/master/Fish/Guppy/src/rolloutEv.py)
