[//]: # (Image References)

[Rewardplot]: convergence_plot.png "Reward over episodes"


# Project 2: Continuous Control

Here I will describe my solution to this project. It was a long, somewhat circuitous path to get to this particular solution. I initially started with ddpg_pendulum's base and started to add on a bunch of features, then try to optimizations. The end result were solutions that were struggling greatly, to converge. After reading [OpenAI's new RL course and it's section on how to debug RL](https://spinningup.openai.com/en/latest/spinningup/exercises.html#problem-set-2-algorithm-failure-modes) I realized that I needed to move back towards the original implementation and make small, carefully vetted changes until I came upon a solution. The solution I'm submitting has the following results, where the objective was reached after 127 episodes. The blue graph shows the average reward across all 20 agents over each episode. The orange graph shows the average of the averaged reward over the past 100 episodes.

![Reward Plot][Rewardplot]

### Learning Algorithm

#### Algorithm

The approach taken in this solution was based on the [DDPG](https://arxiv.org/pdf/1509.02971.pdf) algorithm, as initially implemented in the ddpg_pendulum assignment. In this model we have both an Actor, which is the a neural network representing the *deterministic* policy taking as input the state, and producing the action to take in this state. We also have a Critic, which is a neural network that estimates the Q-value for the policy by taking as input the state and a desired action, and producing the estimated Q-value attained by choosing this action. Much like the approach used in many DQN implementaitons, we use a local and a target copy of the actor and critic, updated at different cadences.

The DDPG algorithm trains over a sequences of independently initialized episodes. At each timestep *t* in the episode we:

1. Use the current *local* actor to determine the next action, and apply noise (in this case Ornstein–Uhlenbeck noise) to the result to encourage exploration.
2. Execute the action, and determine the transition (state, action, reward, next_state, and terminal status of the episode).
3. Store the generated transition in a replay buffer.
4. To avoid correlations between time-steps, randomly sample a set of transitions from the replay buffer.
5. Compute the most recent estimate of the total discounted reward from time *t* by computing the action for the next_state of the transition using the *target* actor, then feeding it into the *target* critic to determine the remaining reward.  That remaining reward is discounted and added to the current reward in the transition.
6. Set the critic loss to be the difference between the above computed estimate of the discounted reward, and the one determined by using the transition's state and action and feeding it to the *local* critic.  This loss is used to perform an update of the *local* critic's neural network.
7. Set the gradient of the policy to be the gradient of the *local* critic, using the *local* actor to determine the action to perform at the transition's state. This loss is negated (to perform gradient *ascent*) and used to update the *local* actor's network.
8. Use a soft update (effectively a linear-interpolation between the *local* and *target* networks) to exponentially update the weights for the *target* networks.

Our changes from the above method include: 
- Using second option with 20 parallel agents to generate more, uncorrelated transitions at each time-step
- Stopping each episode as soon as *one* of the agents terminate
- Using N-step bootstrapping in stage 5, above, to help reduce bias in the training
- Add hyper parameter to determine how frequently to perform learning (stages 4-8) and how many iterations of it to make.

#### Model architectures

The actor network consists of an input layer of the state, followed by a hidden layer of 400 ReLU units, followed by a second hidden layer of 300 ReLU units, followed by a final layer of tanh units (on for each action).

The critic network consists of an input layer of the state, followed by a hidden layer of 400 ReLU units concatenated with the input action, followed by a second hidden layer of 300 ReLU units, followed by a final linear unit of size one which produces the Q-value.

#### Hyperparameters

I did not perform a full-grid search as once I came upon this solution I found convergence quickly. I used N=5 for my N-step bootstrapping as it was the value suggested in [D4PG](https://openreview.net/pdf?id=SyZipzbCb). I also updated one batch of learning, one time for each time-step (thus effectively not using the hyperparameters I added).

128 transition are sampled from the replay buffer, which itself is the default size of 1e5.  The soft-update uses an interpolant of 0.001.  

No weight-decay is used.  While I had intended a grid search with critic learning rates of 1e-3, 1e-4, and 1e-5, and actor learning rates of 1e-4, 1e-5, and 1e-5. I quickly found convergence with both learning-rates set to 1e-4.

### Future Work

I had begun implementing D4PG, specifically the N-step bootstrapping and the [Distributed Prioritized Experience Replay](https://arxiv.org/pdf/1803.00933.pdf). However, while my [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) attained superior rewards earlier, it got increasingly slow over each episode so I terminated it for the time being. I would like to reimplement it using Sumtrees, as they apparently provide superior performance.

I would then like to implement the distributional critic, specifically using [Quantile Regression](https://arxiv.org/pdf/1710.10044) as it provides superior results and can use the Wasserstein Distance to compare distributions which is the "correct" theoretical approach to take.

I would like to explore not terminating all 20 agents until all of them are complete. This may provide more samples, sooner, but it's unclear if the added cost of evaluation would make that benefit or not.

I would like to also explore using [Parameter space noise](https://arxiv.org/pdf/1706.01905) instead of action-space noise.