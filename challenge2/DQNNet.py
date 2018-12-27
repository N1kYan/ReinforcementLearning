import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, hidden_neurons, output):
        #nn.Module.__init__(self)
        super(DQN, self).__init__()
        self.l1 = nn.Linear(5,hidden_neurons)
        self.l2 = nn.Linear(hidden_neurons, output)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x