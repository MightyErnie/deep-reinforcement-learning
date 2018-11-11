from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import time
import ddpg_agent

from cProfile import Profile
from pstats import Stats

temporary_transition_buffer = deque(maxlen=ddpg_agent.REWARD_STEPS)

#env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe')
env = UnityEnvironment(file_name='./Reacher_Windows_x86_64_20/Reacher.exe')
#env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
#env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size
actions = np.zeros((num_agents, action_size))

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

learner = ddpg_agent.Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=10)
actor = ddpg_agent.Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=0)

def tick_simulation(actions):
    global env
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

def tick_learning():
    learner.learn()

def ddpg(n_episodes=2000, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    mean_scores = []
    max_score = -np.Inf
    start_time = time.time()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset()[brain_name]
        states = env_info.vector_observations
        learner.reset()
        score = 0

        for t in range(max_t):
            # Request new actions for each agent
            actions = actor.determine_actions(states)   # TODO: Switch to using the actor
        
            # Step the environment and retrieve the new states, rewards and whether or not the agent is done
            next_states, rewards, dones = tick_simulation(actions)

            # Add this transition to the temporary transition buffer
            temporary_transition_buffer.append((states, actions, np.array(rewards), next_states, np.array(dones)))

            # Store all these transitions into the replay buffer
            if len(temporary_transition_buffer) == ddpg_agent.REWARD_STEPS:
                # Get the initial states, rewards
                states_, actions_, rewards_, _, dones_ = temporary_transition_buffer[0]
                rewards_ *= (1 - dones_)

                # Compute the discounted remainder of the reward
                for step in range(1, ddpg_agent.REWARD_STEPS):
                    gamma = ddpg_agent.GAMMA ** step
                    _, _, reward_, _, done_ = temporary_transition_buffer[step]
                    rewards_ += gamma * reward_ * (1 - done_)

                # Setup the next state info
                _, _, _, next_states_, dones_ = temporary_transition_buffer[-1]

                # Submit the N-step transition to the replay buffer
                for state, action, reward, next_state, done in zip(states_, actions_, rewards_, next_states_, dones_):
                    learner.store_transitions(np.copy(state), np.copy(action), np.copy(reward), np.copy(next_state), np.copy(done))

            # Learn from a minibatch of transitions
            tick_learning()

            # Periodically update the actor
            if (t % ddpg_agent.ACTOR_UPDATE_STEPS) == 0:
                actor.fully_update_actor(learner)

            states = next_states
            score += sum(rewards) / len(rewards)
            if False not in dones:
                break 

        scores_deque.append(score)
        scores.append(score)
        mean_scores.append(np.mean(scores_deque))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tElapsed time: {:.2f}'.format(i_episode, np.mean(scores_deque), score, time.time() - start_time), end="")
        plot_scores(scores, mean_scores)
        if i_episode % 10 == 0:
            torch.save(learner.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(learner.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if (i_episode > 100) and (np.mean(scores_deque) > 30.0):
            torch.save(learner.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(learner.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            return scores
    return scores

# Optional profiling step (just a handful of episodes)
if False:
    profiler = Profile()
    profiler.runcall(ddpg, 60, 700)

    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()

    env.close()
    exit()

scores = ddpg()

plot_scores(scores)

env.close()