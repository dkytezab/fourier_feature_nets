import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy as sp

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: int,
            hidden_width: int,
            activ_fn=nn.ReLU
    ) -> None:

        super(MLP, self).__init__()
        layers = []
        cur_dim = input_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(cur_dim, hidden_width))
            layers.append(activ_fn())
            cur_dim = hidden_width
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(cur_dim, output_dim)

    
    def forward(
            self, 
            input: torch.Tensor
    ) -> torch.Tensor:

        hidden_output = self.hidden_layers(input)
        return self.output_layer(hidden_output)

    
    def train(
            model, 
            train_loader: torch.Tensor, 
            epochs: int, 
            learning_rate=0.005
    ) -> None:

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for step in range(epochs):
            for in_batch, out_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(in_batch)
                loss = torch.mean((outputs - out_batch) ** 2)
                loss.backward()
                optimizer.step()


class FourierFeatureEmbedding():
    def __init__(
            self,
            input_dim: int,
            num_features: int,
            kernel: str,
            length_scale: float
    ) -> None:
        '''Computes random features for embedding'''

        size = (num_features, input_dim)

        if kernel == None:
          random_mat = np.eye(size=size)

        if kernel == 'RBF':
            '''Corresponding spectral kernel is Gaussian with mean=0, variance=1/length_scale**2'''
            random_mat = np.random.normal(loc=0, 
                                          scale=1 / length_scale, 
                                          size=size)
        
        if kernel == 'Laplace':
            '''Corresponding spectral kernel is Cauchy with loc=0, scale=1/length_scale'''
            random_mat = sp.stats.cauchy.rvs(loc=0, 
                                             scale=1 / length_scale, 
                                             size=size)
        self.random_features = random_mat
        self.num_features = num_features

    
    def embed(
            self,
            input: np.ndarray
    ) -> torch.Tensor:
        '''Embeds a given dataset using the computed random features'''

        cos_mat = np.cos(2 * np.pi * self.random_features @ input.T)
        sin_mat = np.sin(2 * np.pi * self.random_features @ input.T)

        embed_mat = np.concatenate(
            (cos_mat, sin_mat), axis=0).T / np.sqrt(self.num_features)
        
        return torch.tensor(embed_mat, dtype=torch.float32)
