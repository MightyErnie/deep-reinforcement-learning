[//]: # (Image References)

[Rewardplot]: score_plot.png "Reward over episodes"


# Project 3: Multiple agents
[D4PG](https://openreview.net/pdf?id=SyZipzbCb)
Here I will describe my solution to this project. My solution was a modification of the solution to project 2's [DDPG](https://arxiv.org/pdf/1509.02971.pdf) based solution, described in detail [here](https://github.com/MightyErnie/deep-reinforcement-learning/blob/master/p2_continuous-control/Report.md). Rather than copy and paste the previous report, this write-up will describe the changes made to handle this multi-agent problem. 

The solution I'm submitting has the following results, where the objective was reached after 1229 episodes. The blue graph shows the max reward over both agents for each episode. The orange graph shows the average of the max reward over the past 100 episodes.  I allowed the training to run long to see how good the results could get.  A 100 episode score of 0.64 was reached after 1579 episodes.

![Reward Plot][Rewardplot]

### Learning Algorithm

#### Algorithm

The training algorithm began as two instances of the actor and critic from [project 2](https://github.com/MightyErnie/deep-reinforcement-learning/blob/master/p2_continuous-control/Report.md), with the critic modified to take, as input, the state of both agents and the actions used by both agents. This was in line with what was described in [Multi Agent Actor Critic for Mixed Cooperative Competitive Environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).

An early bug had the soft-update happening once for each agent.  Correcting this slowed down convergence.  Taking this as a clue, I switched from soft-updates, to a full update delayed to occur once every N steps, as with DQN.

#### Model architectures

The actor network consists of an input layer of the state, followed by a hidden layer of 400 ReLU units, followed by a second hidden layer of 300 ReLU units, followed by a final layer of tanh units (on for each action).

The critic network consists of an input layer of the state of *both* agents, followed by a hidden layer of 400 ReLU units concatenated with the input action of *both* agents, followed by a second hidden layer of 300 ReLU units, followed by a final linear unit of size one which produces the Q-value.

#### Hyperparameters

I used smaller boot-strapping based on an intuition that when the agent fails, there is no recovering, so long bootstrapping woundn't help much. I found N=1 and N=3 seemed to start to improve scores faster than the N=5 used in project #2.  In the end used N=3 for my solution. I also updated one batch of learning, one time for each time-step (thus effectively not using the hyperparameters I added).

128 transition are sampled from the replay buffer, which itself is the default size of 1e5.  The target actors were updated every 10 steps.  

No weight-decay is used.  Convergence was reached with both learning-rates set to 1e-4.

### Future Work

As with project 2, I would like to explore distributional representations for the critic as in [D4PG](https://openreview.net/pdf?id=SyZipzbCb). Further, I would like to explore a singular critic, as the solution space is full observable and both actors are working collaboratively.