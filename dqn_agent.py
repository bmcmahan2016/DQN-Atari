import numpy as np
import random
from collections import namedtuple, deque
import copy
import pdb

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 512         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 1        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        print("using device:", device)
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self._state = deque(maxlen=4)
        self._next_state = deque(maxlen=4)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def reset_state(self, new_state):
        '''
        resets agents internal state
        
        new_state is a 1,84,84,3 RGB image of the world
        '''
        # convert to greyscale
        #new_state = np.expand_dims(np.mean(new_state/255., axis=-1), 0)
        #new_state = new_state.reshape((1,105,2,80,2)).max(4).max(2)
        # new_state is (1, 210, 160)
        # fill deque with current state
        #for i in range(4):
        #    self._state.append(new_state)
        #    self._next_state.append(new_state)
        pass
            
    def update_state_representation(self, curr_state):
        # convert to gray scale
        #curr_state = np.expand_dims(np.mean(curr_state/255., axis=-1), 0)
        #curr_state = curr_state.reshape((1,105,2,80,2)).max(4).max(2)
        # add to history
        #self._state.append(curr_state)
        # convert to 4 frames
        #self.state = np.expand_dims(np.vstack(self._state), 0)
        #print("\n\nagents state shape", self.state.shape, "\n\n")
        pass
        
    def create_next_state(self, next_state):
        # convert to gray scale
        #next_state = np.expand_dims(np.mean(next_state/255., axis=-1), 0)
        #next_state = next_state.reshape((1,105,2,80,2)).max(4).max(2)
        # add to history
        #self._next_state = copy.deepcopy(self._state)
        #self._next_state.append(next_state)
        # convert to 4 frames
        #return np.expand_dims(np.vstack(self._next_state), 0)
        pass
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # convert to gray scale
        #state = np.expand_dims(np.mean(state/255., axis=-1), 0)
        #state = state.reshape((1,105,2,80,2)).max(4).max(2)
        # curr_state is (1, 105, 80)
        # add to memory buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE+5:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #self.update_state_representation(state)

        #state = torch.from_numpy(self.state).float().half().to(device)
        state = torch.from_numpy(state).float().to(device)
        state = torch.unsqueeze(state, 0)  # add batch dimension
 
        # state is (1, 4, 210, 160) or (batch_size, channels, pixels, pixels)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        states = states.float()
        next_states = next_states.float()
        #states = states[:, -1, :]
        #next_states = next_states[:, -1, :]
        #pdb.set_trace()
        # states are (4, 84, 84)
        #states = torch.transpose(states, 1, 3)
        #next_states = torch.transpose(next_states, 1, 3)
        
        #print("states:", states.shape)
        #print("next states:", next_states.shape)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save_model_weights(self):
        """saves Q network weights"""
        self.qnetwork_local.save()
    def load_model_weights(self):
        self.qnetwork_local.load()

# TODO: IMPLEMENT DEQUES FOR EACH ITEM (STATE, ACTION, REWARD, NEXT_STATE, DONE) 
# SO DURING TRAINING BATCHES CAN BE SAMPLED BY LOOPING OVER THESE 5 FIELDS INSTEAD OF LOOPING 
# OVER THE BATCH SIZE WHICH IS MUCH LARGER
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) 

        self.state_memory = torch.zeros((buffer_size, state_size), dtype=torch.float16).to(device) #deque(maxlen=buffer_size) 
        self.action_memory = torch.zeros((buffer_size, 1), dtype=torch.int64).to(device) #deque(maxlen=buffer_size)
        self.reward_memory = torch.zeros((buffer_size, 1)).to(device) #deque(maxlen=buffer_size)
        self.next_state_memory = torch.zeros((buffer_size, state_size), dtype=torch.float16).to(device) #deque(maxlen=buffer_size)
        self.done_memory = torch.zeros((buffer_size, 1)).to(device) #deque(maxlen=buffer_size)

        self.buffer_size = buffer_size
        self.replay_ix = 0  # counts where in the replay buffer we are
        self.buffer_full = False

        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        #e = self.experience(state, action, reward, next_state, done)
        #self.memory.append(e)
        # new implementation below
        buffer_ix = self.replay_ix %self.buffer_size
        self.state_memory[buffer_ix] = torch.from_numpy(state)
        self.action_memory[buffer_ix] = action
        self.reward_memory[buffer_ix] = reward
        self.next_state_memory[buffer_ix] = torch.from_numpy(next_state)
        self.done_memory[buffer_ix] = done
        self.replay_ix += 1
        if self.replay_ix %self.buffer_size == 0 and not self.buffer_full:
            self.buffer_full = True
            print("replay buffer full")
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        # we need at least 4 future frames to construct the next_state
        batch_ixs = np.random.choice(self.__len__(), self.batch_size)
        #print("batch indices:", batch_ixs)
        #pdb.set_trace()
        states = self.state_memory[batch_ixs]#torch.stack((states1, states2, states3, states4), dim=1)  # (batch_size, 4, pixels, pixels)
        next_states = self.next_state_memory[batch_ixs]#torch.stack((next_states_1, next_states_2, next_states_3, next_states_4), dim=1)  # (batch_size, 4, pixels, pixels)

        #states = torch.from_numpy(states).float().to(device)
        actions = self.action_memory[batch_ixs]
        #actions = torch.from_numpy(actions).long().to(device)
        rewards = self.reward_memory[batch_ixs]
        #rewards = torch.from_numpy(rewards).float().to(device)
        #next_states = self.next_state_memory[batch_ixs]
        #next_states = torch.from_numpy(next_states).float().to(device)
        dones = self.done_memory[batch_ixs]
        #dones = torch.from_numpy(dones).float().to(device)

        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        if self.buffer_full:
            return self.buffer_size
        return self.replay_ix