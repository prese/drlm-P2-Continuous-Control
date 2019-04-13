
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

I've chosen to solve [Reacher 20](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

![Trained Agent][image1]

The goal of the agent is to maintain its position at the target location for as many time steps as possible. The environment contains 20 identical agents, each with its own copy of the environment.



### Learning Algorithm


My tested environment has 20 different agents. Because all of the agents perform the same task under the same conditions, I've used a single brain to control all the 20 agents. The action space is continuous and the  `value-based method`  like DQN is not suitable in this environment.  So I decided to use the  `policy-based method`.

Policy-based methods offer practical ways of dealing with large actions spaces, even continuous spaces with an infinite number of actions. Instead of computing learned probabilities for each of the many actions, we instead learn statistics of the probability distribution.

I trained the network using  `Deep Deterministic Policy Gradients(DDGP)`  algorithm. It is an  `Actor-Critic method`  in which two architectures are combined. Actor determines the current policy in continuous space and Critic learns the Q-values in a given (state, action) pair.





#### Model architecture

In my implementation the Actor network contains two hidden layers of 256 devices with ReLU activation applied to both layers and a tanh on the end.

In addition, a batch normalization is applied to the input and between the hidden layers.

The Critical network also has two hidden layers with 256 units and ReLU Activation on both layers.


#### DDPG hyper-parameters
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-5     # L2 weight decay
```
The algorithm works by having the agent act in the environment and collecting rewards. These experiences are then stored in a buffer and used at a later time to train the neural networks. A neural network, called actor network is used to approximate the optimal policy from which the best believed action is selected. During training another neural network, called the critic network is used to estimate value function which is then used to evaluate the action selected by the actor network. By using these two networks in tandem, the agent is able to solve complex environments that contain both a continuous state and action space.  An Ornstein-Uhlenbeck process is used to generate random noise which is then scaled by a decaying epsilon value.



### Plots of rewards

### Ideas for improvements

 - Change network sizes and choose different hyperparameters
 - Trying other algorithms like PPO, A3C or D4PG
