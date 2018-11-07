from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

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
actions = np.zeros((num_agents, action_size))

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

from ddpg_agent import Agent
agents = [Agent(state_size=state_size, action_size=action_size, random_seed=10 + i) for i in range(num_agents)]

def ddpg(n_episodes=200, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset()[brain_name]
        states = env_info.vector_observations
        for agent in agents:
            agent.reset()
        score = 0
        for t in range(max_t):
            for idx, (agent, state) in enumerate(zip(agents, states)):
                actions[idx] = agent.act(state)
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for idx, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
                agents[idx].step(np.copy(state), np.copy(action), np.copy(reward), np.copy(next_state), np.copy(done))

            states = next_states
            score += sum(rewards) / len(rewards)
            if False not in dones:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("score_plot.png", dpi=600)

env.close()