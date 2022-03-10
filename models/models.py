import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_layers, input_dim):
        super().__init__()

        layers = []
        for layer_idx in range(n_layers):
            output_dim = input_dim//2
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
            input_dim = output_dim

        layers.extend([nn.Linear(input_dim, 10), nn.Softmax(dim=1)])

        self.model = nn.Sequential(*layers)
    
    def forward(self, img):
        return self.model(img)
