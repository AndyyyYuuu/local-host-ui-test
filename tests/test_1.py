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

valid_loss_graph = trainreaction.Line("Valid Loss")
valid_loss_graph.color("green")

progress_bar_1 = trainreaction.Bar("Progress 1")
progress_bar_1.color("blue")

progress_bar_2 = trainreaction.Bar("Progress 2")
progress_bar_2.color("yellow")

for i in range(100):
    train_loss_graph.update(math.sin(i) + 1)
    valid_loss_graph.update(math.cos(i) + 1)
    progress_bar_1.update(i / 9)
    progress_bar_2.update(i / 10)
    trainreaction.send_lm_message("Hello "*i)
    time.sleep(2)

