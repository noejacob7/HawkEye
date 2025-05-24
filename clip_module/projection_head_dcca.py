import torch
import torch.nn as nn

class DCCAProjectionHead(nn.Module):
    """
    A Deep CCA-style projection head to map CLIP text embeddings
    into LightMatchNet embedding space using hidden layers.
    """
    def __init__(self, input_dim=512, output_dim=128, hidden_dims=[512, 256]):
        super(DCCAProjectionHead, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.projector(x)
