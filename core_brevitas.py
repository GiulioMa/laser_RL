import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
from torch.distributions import Normal

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if torch.is_tensor(shape) or isinstance(shape, int) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([torch.prod(torch.tensor(p.shape)) for p in module.parameters()])

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianQuantMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super(SquashedGaussianQuantMLPActor, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=14, return_quant_tensor=True)
        self.fc1 = QuantLinear(obs_dim, hidden_sizes[0], bias=True, 
                               weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=True)
        self.fc1.cache_inference_quant_bias=True
        self.relu1 = QuantReLU(bit_width=23, return_quant_tensor=True)
        self.fc2 = QuantLinear(hidden_sizes[0], hidden_sizes[1], bias=True, 
                               weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=True)
        self.fc2.cache_inference_quant_bias=True
        self.relu2 = QuantReLU(bit_width=37, return_quant_tensor=True)
        self.fc3 = QuantLinear(hidden_sizes[1], 2 * act_dim, bias=True, 
                               weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=True)
        self.fc3.cache_inference_quant_bias=True
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        obs = self.quant_inp(obs)
        net_out = self.fc1(obs)
        net_out = self.relu1(net_out)
        net_out = self.fc2(net_out)
        net_out = self.relu2(net_out)
        net_out = self.fc3(net_out)

        # Extract the underlying floating-point values
        net_out = net_out.value
        
        # Split output into mu and log_std
        mu, log_std = torch.chunk(net_out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (torch.log(torch.tensor(2.0)) - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianQuantMLPActor(obs_dim, act_dim, hidden_sizes, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a
