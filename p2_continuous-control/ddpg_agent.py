import numpy as np
from numpy_ringbuffer import RingBuffer
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

STARTING_BETA = 0.4
BETA_STEPS = BUFFER_SIZE

REWARD_STEPS = 4

ACTOR_UPDATE_STEPS = 3

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
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(num_agents, action_size, random_seed)
        
        # Prioritized replay annealing
        self.beta = STARTING_BETA
        self.beta_step = (1.0 - STARTING_BETA) / BETA_STEPS
    
        # Replay memory
        self.memory = PrioritizedReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def store_transitions(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def determine_actions(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

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

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + ((GAMMA ** REWARD_STEPS) * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)

        # Compute weighted losses for the prioritized replay
        loss = (Q_targets - Q_expected) ** 2.0
        if (weights is not None) and (sample_indices is not None):
            self.memory.update_priorities(sample_indices, torch.abs(Q_targets - Q_expected).detach().cpu().data.numpy())

            weights = weights.unsqueeze(-1)
            weighted_loss = (weights * loss)

        # Compute critic loss
        critic_loss = loss.mean()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

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
        dx = self.theta * (self.mu - x) + self.sigma * np.random.random(self.state.shape)
        self.state = x + dx
        return self.state

class ReplayBufferDeque:
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), None, None

    def update_priorities(self, indices, new_priorities):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

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
        self.buffer_size = buffer_size
        self.used_size = 0
        self.next_entry = 0

        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # ANIS TODO: Handle adding batches at once
        self.states[self.next_entry] = state
        self.actions[self.next_entry] = action
        self.rewards[self.next_entry] = reward
        self.next_states[self.next_entry] = next_state
        self.dones[self.next_entry] = done

        self.next_entry = (self.next_entry + 1) % self.buffer_size
        self.used_size = min(self.used_size + 1, self.buffer_size)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        sample_indices = np.random.choice(self.used_size, self.batch_size)

        states = torch.from_numpy(self.states[sample_indices]).float().to(device)
        actions = torch.from_numpy(self.actions[sample_indices]).float().to(device)
        rewards = torch.from_numpy(self.rewards[sample_indices]).float().to(device).unsqueeze(-1)
        next_states = torch.from_numpy(self.next_states[sample_indices]).float().to(device)
        dones = torch.from_numpy(self.dones[sample_indices].astype(np.uint8)).float().to(device).unsqueeze(-1)

        return (states, actions, rewards, next_states, dones), None, None

    def update_priorities(self, indices, new_priorities):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return self.used_size


class PrioritizedReplayBufferDeque:
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
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = 0.6
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        if (len(self.priorities) > 0):
            self.priorities.append(max(self.priorities))
        else:
            self.priorities.append(1.0)
    
    def sample(self, beta):
        # Compute the normalized probabilities
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= sum(probabilities)

        sample_indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)

        states = torch.from_numpy(np.vstack([self.memory[i].state for i in sample_indices])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in sample_indices])).float().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in sample_indices])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in sample_indices])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in sample_indices]).astype(np.uint8)).float().to(device)

        weights = (len(probabilities) * probabilities[sample_indices]) ** (-beta)
        weights /= weights.max()

        return (states, actions, rewards, next_states, dones), sample_indices, weights

    def update_priorities(self, indices, new_priorities):
        new_priorities = new_priorities.reshape(new_priorities.shape[0])
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.buffer_size = buffer_size
        self.used_size = 0
        self.next_entry = 0

        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.priorities = np.ones(buffer_size, dtype=np.float64)

        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.alpha = 0.6
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # ANIS TODO: Handle adding batches at once
        self.states[self.next_entry] = state
        self.actions[self.next_entry] = action
        self.rewards[self.next_entry] = reward
        self.next_states[self.next_entry] = next_state
        self.dones[self.next_entry] = done

        if self.used_size != 0:
            self.priorities[self.next_entry] = np.max(self.priorities[:self.used_size])

        self.next_entry = (self.next_entry + 1) % self.buffer_size
        self.used_size = min(self.used_size + 1, self.buffer_size)

    def generate_sampling_probabilities(self):
        probabilities = np.power(self.priorities[:self.used_size], self.alpha)
        probabilities /= np.sum(probabilities)
        return probabilities
    
    def sample(self, beta):
        # Compute the normalized probabilities
        probabilities = self.generate_sampling_probabilities()

        sample_indices = np.random.choice(self.used_size, self.batch_size, p=probabilities)

        states = torch.from_numpy(self.states[sample_indices]).float().to(device)
        actions = torch.from_numpy(self.actions[sample_indices]).float().to(device)
        rewards = torch.from_numpy(self.rewards[sample_indices]).float().to(device).unsqueeze(-1)
        next_states = torch.from_numpy(self.next_states[sample_indices]).float().to(device)
        dones = torch.from_numpy(self.dones[sample_indices].astype(np.uint8)).float().to(device).unsqueeze(-1)

        weights = torch.from_numpy(self.used_size * probabilities[sample_indices]).float().to(device)
        weights = torch.pow(weights, -beta)
        weights /= torch.max(weights)

        return (states, actions, rewards, next_states, dones), sample_indices, weights

    def update_priorities(self, indices, new_priorities):
        new_priorities = new_priorities.reshape(new_priorities.shape[0])
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """Return the current size of internal memory."""
        return self.used_size
