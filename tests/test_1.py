import brainscan
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
brainscan.init()

while True:
    time.sleep(2)
    print("*")

    brainscan.graph_value("loss", random.randint(0, 100))

