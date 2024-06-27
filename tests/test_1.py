import trainreaction
import torch
from torch import nn
import random
import time


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


model = MLP(3, 4, 1)
trainreaction.init()

while True:

    trainreaction.update_graph("Train Loss", random.uniform(0, 1))
    trainreaction.update_graph("Valid Loss", random.uniform(0, 1))
    time.sleep(2)
