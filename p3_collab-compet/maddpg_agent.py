import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

PER_EPSILON = 1e-5
STARTING_BETA = 0.4
BETA_STEPS = BUFFER_SIZE

REWARD_STEPS = 1

LEARN_STEPS = 1
LEARN_INTERVAL = 1
ACTOR_UPDATE_INTERVAL = 1

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_optimizer = [optim.Adam(self.actor_local[agent].parameters(), lr=LR_ACTOR) for agent in range(num_agents)]

        # Critic Network (w/ Target Network)
        self.critic_local = [Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device) for _ in range(num_agents)]
        self.critic_target = [Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device) for _ in range(num_agents)]
        self.critic_optimizer = [optim.Adam(self.critic_local[agent].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for agent in range(num_agents)]

        # Noise process
        self.noise = OUNoise(num_agents, action_size, random_seed)
        
        # Prioritized replay annealing
        self.beta = STARTING_BETA
        self.beta_step = (1.0 - STARTING_BETA) / BETA_STEPS

        # Transition buffer for the N-lookahead
        self.transition_buffer = deque(maxlen=REWARD_STEPS)
    
        # Replay memory
        self.memory = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def store_transitions(self, states, actions, rewards, next_states, dones):
        # Add this transition to the temporary transition buffer
        self.transition_buffer.append((states, actions, np.array(rewards), next_states, np.array(dones)))

         # Store all these transitions into the replay buffer
        if len(self.transition_buffer) == REWARD_STEPS:
            # Get the initial states, rewards
            states_, actions_, rewards_, _, dones_ = self.transition_buffer[0]
            rewards_ *= (1 - dones_)

            # Compute the discounted remainder of the reward
            for step in range(1, REWARD_STEPS):
                gamma = GAMMA ** step
                _, _, reward_, _, done_ = self.transition_buffer[step]
                rewards_ += gamma * reward_ * (1 - done_)

            # Setup the next state info
            _, _, _, next_states_, dones_ = self.transition_buffer[-1]

            # Save experience / reward
            self.memory.add(states, actions, rewards, next_states, dones)

    def determine_actions(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = np.empty((self.num_agents, self.action_size), dtype=np.float32)
        for agent in range(self.num_agents):
            state = torch.from_numpy(states[agent]).float().to(device)
            self.actor_local[agent].eval()
            with torch.no_grad():
                actions[agent] = self.actor_local[agent](state).cpu().data.numpy()
            self.actor_local[agent].train()
            if add_noise:
                actions[agent] += self.noise.sample()[agent]

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """

        # Learn, if enough samples are available in memory
        if len(self.memory) < BATCH_SIZE:
            return

        (states, actions, rewards, next_states, dones), sample_indices, weights = self.memory.sample(self.beta)
        self.beta += self.beta_step
        self.beta = max(self.beta, 1.0)

        # Determine the next actions
        actions_next = torch.stack([self.actor_target[agent](next_states[:, agent]) for agent in range(self.num_agents)], dim=1)

        for agent in range(self.num_agents):
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            Q_targets_next = self.critic_target[agent](next_states.view(BATCH_SIZE, -1), actions_next.view(BATCH_SIZE, -1))

            # Compute Q targets for current states (y_i)
            Q_targets = rewards[:, agent].unsqueeze(-1) + ((GAMMA ** REWARD_STEPS) * Q_targets_next * (1 - dones[:, agent].unsqueeze(-1)))
            Q_expected = self.critic_local[agent](states.view(BATCH_SIZE, -1), actions.view(BATCH_SIZE, -1))

            # Compute weighted losses for the prioritized replay
            critic_loss = None
            if (weights is not None) and (sample_indices is not None):
                self.memory.update_priorities(sample_indices, torch.abs(Q_targets - Q_expected).detach().cpu().data.numpy())

                weights = torch.sqrt(weights.unsqueeze(-1))
                critic_loss = F.mse_loss(weights * Q_targets, weights * Q_expected)
            else:
                # Compute critic loss
                critic_loss = F.mse_loss(Q_targets, Q_expected)

            # Minimize the loss
            self.critic_optimizer[agent].zero_grad()
            critic_loss.backward(retain_graph=agent == 0)
            self.critic_optimizer[agent].step()

        # Determine the predicted actions
        actions_pred = torch.stack([self.actor_local[agent](states[:, agent]) for agent in range(self.num_agents)], dim=1)

        for agent in range(self.num_agents):
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actor_loss = -self.critic_local[agent](states.view(BATCH_SIZE, -1), actions_pred.view(BATCH_SIZE, -1)).mean()

            # Minimize the loss
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward(retain_graph=agent == 0)
            self.actor_optimizer[agent].step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local[agent], self.critic_target[agent], TAU)
            self.soft_update(self.actor_local[agent], self.actor_target[agent], TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_targets(self):
        """Update model parameters.
        """
        for target_param, local_param in zip(critic_target.parameters(), critic_local.parameters()):
            target_param.data.copy_(local_param.data)
        for target_param, local_param in zip(actor_target.parameters(), actor_local.parameters()):
            target_param.data.copy_(local_param.data)

    def update_actor_from_learner(self, agent):
        """Copies actor networks from the source

        Params
        ======
            agent: Agent providing the new network
        """
        for target_param, source_param in zip(self.actor_local.parameters(), agent.actor_local.parameters()):
            target_param.data.copy_(source_param)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, num_agents, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones((num_agents, size))
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.state.shape)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), None, None

    def update_priorities(self, indices, new_priorities):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
