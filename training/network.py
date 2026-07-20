from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
