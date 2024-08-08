import trainreaction
import torch
from torch import nn
import random
import time
import math


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

train_loss_graph = trainreaction.Line("Train Loss")
train_loss_graph.color("green")

progress_bar = trainreaction.Bar("Progress")

for i in range(10):
    train_loss_graph.update(math.sin(i)+1)
    progress_bar.update(i/9)
    time.sleep(2)

