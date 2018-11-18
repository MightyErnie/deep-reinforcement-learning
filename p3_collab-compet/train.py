from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import maddpg_agent
import torch

TARGET_SCORE = 0.5

env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

def tick_simulation(actions):
    env_info = env.step(actions)[brain_name]
    return env_info.vector_observations, env_info.rewards, env_info.local_done

def plot_scores(scores, mean_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.plot(np.arange(1, len(mean_scores)+1), mean_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("score_plot.png", dpi=600)
    plt.close(fig)

def maddpg(n_episodes=2500, max_t=10000):
    agent = maddpg_agent.Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=31337)
    
    best_mean = 0.0
    scores_deque = deque(maxlen=100)
    scores = []
    windowed_mean_scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)

        for t in range(max_t):
            # Request new actions for each agent
            actions = agent.determine_actions(states)

            # Step the environment and retrieve the new states, rewards and whether or not the agent is done
            next_states, rewards, dones = tick_simulation(actions)

            # Submit the N-step transition to the replay buffer
            agent.store_transitions(states, actions, rewards, next_states, dones)

            # Learn from a minibatch of transitions
            if (t % maddpg_agent.LEARNING_INTERVAL) == 0:
                for _ in range(maddpg_agent.LEARNING_STEPS):
                    agent.learn()

            if (t % maddpg_agent.ACTOR_UPDATE_INTERVAL) == 0:
                agent.update_targets()
            
            states = next_states
            score += rewards
            if np.any(dones):
                break 

        agent_max_score = np.max(score)

        scores_deque.append(agent_max_score)
        scores.append(agent_max_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\t'.format(i_episode, np.mean(scores_deque), agent_max_score))

        windowed_mean_scores.append(np.mean(scores_deque))
        if (i_episode % 100) == 0:
            plot_scores(scores, windowed_mean_scores)
        mean_recent_scores = np.mean(scores_deque)
        if (i_episode > 100) and (mean_recent_scores > TARGET_SCORE):
            if mean_recent_scores > best_mean:
                best_mean = mean_recent_scores
                torch.save(agent.actor_local[0].state_dict(), 'checkpoint_actor0.pth')
                torch.save(agent.critic_local[0].state_dict(), 'checkpoint_critic0.pth')
                torch.save(agent.actor_local[1].state_dict(), 'checkpoint_actor1.pth')
                torch.save(agent.critic_local[1].state_dict(), 'checkpoint_critic1.pth')
                print("New best score of {:.2f}".format(mean_recent_scores))
    return scores

scores = maddpg()

env.close()