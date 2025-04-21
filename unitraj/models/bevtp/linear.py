import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(in_dim, hidden_dim, out_dim, dropout=0.1):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim)
    )
    return mlp


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class LayerScale(nn.Module):
    def __init__(self, dim, scale=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x):
        return x * self.weight


class FFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, num_fcs=2, dropout=0.1, layer_scale_init_value=1e-5):
        super().__init__()
        assert num_fcs >= 2, 'num_fcs should be no less than 2'

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = feedforward_channels

        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

        self.layer_scale = LayerScale(embed_dims, scale=layer_scale_init_value)

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.layer_scale(out)

        if identity is None:
            identity = x
        return identity + out
    
    
def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MotionRegHead(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''

    def __init__(self, D=64):
        super().__init__()
        self.D = D
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(D, D)), nn.ReLU(),
            init_(nn.Linear(D, D)), nn.ReLU(),
            init_(nn.Linear(D, 5))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        pred_obs = self.observation_model(agent_decoder_state)

        x_mean = pred_obs[..., 0]
        y_mean = pred_obs[..., 1]
        x_sigma = F.softplus(pred_obs[..., 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[..., 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[..., 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=-1)
    
    
class MotionClsHead(nn.Module):
    def __init__(self, D=64, D_T=64, future_len=12):
        super().__init__()
        self.D = D
        self.D_T = D_T
        self.T = future_len
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.ModuleList([
            nn.Sequential(init_(nn.Linear(D, D_T)), nn.ReLU()),
            nn.Sequential(init_(nn.Linear(D_T*self.T, D)), nn.ReLU()),
            init_(nn.Linear(D, 1))
        ])
    
    def forward(self, agent_decoder_state): # (K, B, T, D)
        K, B, T, D = agent_decoder_state.shape
        agent_decoder_state = self.observation_model[0](agent_decoder_state).reshape(K, B, -1)
        agent_decoder_state = self.observation_model[1](agent_decoder_state)
        pred_obs = self.observation_model[2](agent_decoder_state)
        
        return pred_obs
        
        
        
