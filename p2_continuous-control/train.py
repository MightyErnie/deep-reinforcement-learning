from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import time
import ddpg_agent
import sys

from cProfile import Profile
from pstats import Stats

TARGET_SCORE = 30.0

env = UnityEnvironment(file_name='./Reacher_Windows_x86_64_20/Reacher.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

agent = ddpg_agent.Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=31337)

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

def ddpg(n_episodes=500, max_t=10000):
    scores_deque = deque(maxlen=100)
    scores = []
    windowed_mean_scores = []
    start_time = time.time()
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
            if (t % ddpg_agent.LEARN_INTERVAL) == 0:
                for _ in range(ddpg_agent.LEARN_STEPS):
                    agent.learn()
            
            states = next_states
            score += rewards
            if np.any(dones):
                break 

        agent_ave_score = np.mean(score)

        scores_deque.append(agent_ave_score)
        scores.append(agent_ave_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tElapsed time: {:.2f}'.format(i_episode, np.mean(scores_deque), agent_ave_score, time.time() - start_time))

        windowed_mean_scores.append(np.mean(scores_deque))
        plot_scores(scores, windowed_mean_scores)
        if (i_episode > 100) and (np.mean(scores_deque) > TARGET_SCORE):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            return scores
    return scores

# Optional profiling step (just a handful of episodes)
if len(sys.argv) > 1 and sys.argv[1].lower() == 'profile':
    ddpg(70, 700)

    env.close()
    exit()

scores = ddpg()

env.close()