import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        self._action_size = action_size
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.val_fc1 = nn.Linear(fc1_units, 1)
        self.adv_fc1 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))


        # seperate input into two streams
        value_stream = self.val_fc1(x)
        advantage_stream = self.adv_fc1(x)

        # must transpose terms for addition broadcasting 
        # V(s): (batch_size, 1) --> (1, batch_size)
        V_s = torch.transpose(value_stream, 0, 1)

        # A(s,a): (batch_size, 4) --> (4, batch_size)
        A_sa = torch.transpose(advantage_stream, 0, 1)
        A_sa = A_sa - (1/self._action_size)*torch.sum(A_sa, 0)

        Q_vals = V_s - A_sa

        # Q(s,a): (4, batch_size) --> (batch_size, 4)
        Q_vals = torch.transpose(Q_vals, 0, 1)
        return Q_vals


        return self.fc3(x)
    
    def save(self):
        """saves model weights"""
        torch.save(self.state_dict(), "model.pt")
    
    def load(self):
        """loads model weights"""
        self.load_state_dict(torch.load("model.pt"))
        
        
'''import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self._action_size = action_size
        self.seed = torch.manual_seed(seed)
        #self.conv1 = nn.Conv2d(4, 32, (8,8), stride=4, dtype=torch.float16)
        #self.conv2 = nn.Conv2d(32, 64, (4,4), stride=2, dtype=torch.float16)
        #self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1, dtype=torch.float16)
        self.fc1 = nn.Linear(state_size, 64, dtype=torch.float16)
        #nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 64, dtype=torch.float16)
        #nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(64, 1, dtype=torch.float16)
        #nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(64, action_size, dtype=torch.float16)
        self.fc5 = nn.Linear(action_size, action_size, dtype=torch.float16)

    def forward(self, state):
        """
        Build a network that maps state -> action values.

        Parameters:
        -state is (batch_size, 37) and contains a 37 dimensional beam based perception
            of the environment in front of the agent.
        """
        # state is (batch_size, 3, 84, 84)
        #print("input to model:", state.shape)
        #X = F.relu(self.conv1(state))
        #X = F.relu(self.conv2(X))
        #X = F.relu(self.conv3(X))
        
        
        
        
        #X = torch.flatten(X, start_dim=1)
        #print("\n\nflattened shape:", X.shape, "\n\n")
        X = F.relu(self.fc1(state))
        X = F.relu(self.fc2(X))

        # seperate input into two streams
        value_stream = self.fc3(X)
        advantage_stream = F.relu(self.fc4(X))
        advantage_stream = self.fc5(advantage_stream)

        # must transpose terms for addition broadcasting 
        # V(s): (batch_size, 1) --> (1, batch_size)
        V_s = torch.transpose(value_stream, 0, 1)

        # A(s,a): (batch_size, 4) --> (4, batch_size)
        A_sa = torch.transpose(advantage_stream, 0, 1)
        A_sa = A_sa - (1/self._action_size)*torch.sum(A_sa, 0)

        Q_vals = V_s - A_sa

        # Q(s,a): (4, batch_size) --> (batch_size, 4)
        Q_vals = torch.transpose(Q_vals, 0, 1)
        return Q_vals
    
    def save(self):
        """saves model weights"""
        torch.save(self.state_dict(), "model.pt")
    
    def load(self):
        """loads model weights"""
        self.load_state_dict(torch.load("model.pt"))
        
        
'''