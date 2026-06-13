import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPOAgent(nn.Module):
    def __init__(self, num_servers, hidden_size=64):
        super(PPOAgent, self).__init__()
        self.num_servers = num_servers
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.num_servers, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_policy = nn.Linear(self.hidden_size, self.num_servers)
        self.fc_value = nn.Linear(self.hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc_value(x)

    def get_action(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy_dist = torch.softmax(self.fc_policy(x), dim=-1)
        dist = torch.distributions.Categorical(policy_dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy_dist = torch.softmax(self.fc_policy(x), dim=-1)
        dist = torch.distributions.Categorical(policy_dist)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        value = self.fc_value(x)
        return log_prob, value, entropy
