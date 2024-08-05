import scipy.signal
import os
import numpy as np
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int8Bias
from torch.distributions import Normal
from brevitas.quant_tensor import QuantTensor

START_EPISODE = 50
EPISODE_PER_EPOCH = 10

N_STEPS = 81
N_INPUT = 2
N_OUTPUT = 2
N_HIDDEN_1 = 64
N_HIDDEN_2 = 80
ACT_LIM = 10000


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

scale = 12000.

def quantize_input_tensor(in_float, scale = 1/scale, bit_width = 16, zero_point = 0.0, training=False, device=torch.device('cpu')):
    int_value = in_float * scale
    quant_value = (int_value - zero_point) * scale
    quant_tensor_input = QuantTensor(
    quant_value,
    scale=torch.tensor(scale),
    zero_point=torch.tensor(zero_point),
    bit_width=torch.tensor(float(bit_width)),
    signed=True,
    training=training).to(device)

    return quant_tensor_input

class Digital_twin(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, device):
        super(Digital_twin, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=16, return_quant_tensor=True, narrow_range=False, signed=True)
        self.fc1 = QuantLinear(obs_dim, hidden_sizes[0], bias=True, 
                               weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=True)
        self.fc1.cache_inference_quant_bias=True
        self.relu1 = QuantReLU(act_quant=None, return_quant_tensor=True)
        self.fc2 = QuantLinear(hidden_sizes[0], hidden_sizes[1], bias=True, 
                               weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=True)
        self.fc2.cache_inference_quant_bias=True
        self.relu2 = QuantReLU(act_quant=None, return_quant_tensor=True)
        self.fc3 = QuantLinear(hidden_sizes[1], 2 * act_dim, bias=True, 
                               weight_quant=Int8WeightPerTensorFloat, bias_quant=Int8Bias, return_quant_tensor=True)
        self.fc3.cache_inference_quant_bias=True
        self.device = device
        
    def forward(self, obs):
        obs = quantize_input_tensor(obs, training=self.training, device=self.device)
        # obs = self.quant_inp(obs)
        ##print(obs)
        net_out = self.fc1(obs)
        #print(net_out)
        net_out = self.relu1(net_out)
        #print(net_out)
        net_out = self.fc2(net_out)
        #print(net_out)
        net_out = self.relu2(net_out)
        #print(net_out)
        net_out = self.fc3(net_out)
        #print(net_out)
        
        return net_out
        
        
class SquashedGaussianQuantMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit, device):
        super(SquashedGaussianQuantMLPActor, self).__init__()
        self.twin = Digital_twin(obs_dim, act_dim, hidden_sizes, device)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.twin(obs)

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
        pi_action = self.act_limit * pi_action/ACT_LIM

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, device, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = N_INPUT
        act_dim = 1
        act_limit = ACT_LIM

        # build policy and value functions
        self.pi = SquashedGaussianQuantMLPActor(obs_dim, act_dim, hidden_sizes, act_limit, device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), os.path.join(dir_path, 'model.pth'))

    @classmethod
    def load(cls, dir_path, device):
        model = cls(device)
        model.load_state_dict(torch.load(os.path.join(dir_path, 'model.pth'), map_location=device))
        return model

def receive_data_episode(client, num_data_elem, num_action_elem):
        num_bytes = num_data_elem * 2 + num_action_elem * 4  # Each data is int16, each action is a float
        packet_size = 1024 * 32 # packet size
        data_buffer = bytearray()

        while len(data_buffer) < num_bytes:
            data = client.recv(packet_size)
            if not data:
                print("Connection closed by server or error occurred.")
                break
            data_buffer.extend(data)

        # Split into samples and actions 
        samples = data_buffer[:num_data_elem * 2]
        actions = data_buffer[num_data_elem * 2:]

        #print(f'Number of received actions: {len(actions)}')

        sample1 = []
        sample2 = []
        for i in range(0, len(samples), 4):  # Step by 4 bytes to read two int16_t
            s1, s2 = struct.unpack('<hh', samples[i:i+4])
            sample1.append(s1/scale)#12000.
            sample2.append(s2/scale)#12000.

        # List to hold the resulting lists of actions
        action_list = []
        
        # Iterate over the byte string in chunks of 16 bytes
        for i in range(0, len(actions), 4):
            # Extract a 4-byte segment
            segment = actions[i:i+4]
            # Unpack the segment into a float (little-endian)
            action = struct.unpack('<f', segment)[0]
            # Unnormalize the action (to have it in range (-ACT_LIM, ACT_LIM))
            action = 2*action - ACT_LIM
            # Append the action to the action_list
            action_list.append(action/ACT_LIM)
            
        obs = []
        for step in range(N_STEPS):
            o = []
            for i in range(N_INPUT//2):
                o.append(sample1[step * N_INPUT//2 + i])
                o.append(sample2[step * N_INPUT//2 + i])
            o = np.array(o, dtype=np.float32) #.reshape(1, N_INPUT)
            obs.append(o)    
            
        return obs, action_list