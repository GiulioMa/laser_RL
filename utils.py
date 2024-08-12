import core_FPGA as core
import numpy as np
import random
import torch
import struct

def create_input_tensor(dim, scale, device):
    """
    Creates a tensor with shape (1, dim) containing uniformly distributed random numbers 
    within the range of int16.

    Parameters:
    dim (int): The dimension of the tensor to be created.

    Returns:
    torch.Tensor: A tensor of shape (1, dim) with uniformly distributed random numbers 
                  ranging from the minimum to the maximum value of int16.
    
    Example:
    >>> create_input_tensor(10)
    tensor([[ 21426.9727, -15413.4648,  31353.7070, -26877.7969, -31792.7910,
              2873.2275,   9894.8887,  -6331.2305, -22668.4844,  25723.5293]])
    """
    # Define the range for int16
    min_int16 = np.iinfo(np.int16).min
    max_int16 = np.iinfo(np.int16).max
    
    # Generate random numbers uniformly distributed 
    random_numbers = np.random.uniform(low=10., high=16000., size=(1, dim)).astype(np.int16)    
    # Convert the random numbers to a tensor
    tensor = torch.tensor(random_numbers, dtype=torch.float32, device=device)
    
    return tensor * scale

def weights_creation(model, scale, ep_num):
    """
    Generates a buffer of weights and random numbers for a given model and episode number.

    This function extracts and processes the quantized weights and biases from the provided
    model's layers (fc1, fc2, fc3) and generates a list of random numbers based on the 
    episode number. The function returns the combined weights and random numbers as a list of 
    int8 values.

    Parameters:
    model (torch.nn.Module): The neural network model with quantized weights and biases.
    scale (float): A scale factor to be included in the weights buffer.
    ep_num (int): The current episode number to determine the random number generation method.

    Returns:
    tuple: A tuple containing:
        - weights_buffer (list of int): The buffer containing weights, biases, scale, and random numbers.
        - random_numbers (list of float): The list of generated random numbers.
    """
    model.eval()
    # Store the random numbers
    random_numbers = []
    
    # Create a weights buffer
    weights_buffer = []
    
    # Take care of W1
    W1 = model.fc1.quant_weight().int().cpu().numpy().T.flatten()
    #print(max(W1))
    for j in range(core.N_INPUT):
        for i in range(core.N_HIDDEN_1):
            weights_buffer.append(W1[i + j * core.N_HIDDEN_1])
    
    # Taking care of b1
    b1 = model.fc1.quant_bias().int().cpu().numpy().flatten()
    #print(max(b1))
    weights_buffer.extend(b1)
    
    # Taking care of W2
    W2 = model.fc2.quant_weight().int().cpu().numpy().T.flatten()
    #print(max(W2))
    for j in range(core.N_HIDDEN_1):
        for i in range(core.N_HIDDEN_2):
            weights_buffer.append(W2[i + j * core.N_HIDDEN_2])
    
    # Taking care of b2
    b2 = model.fc2.quant_bias().int().cpu().numpy().flatten()
    #print(max(b2))
    weights_buffer.extend(b2)
    
    # Taking care of W3
    W3 = model.fc3.quant_weight().int().cpu().numpy().T.flatten()
    #print(max(W3))
    for j in range(core.N_HIDDEN_2):
        for i in range(core.N_OUTPUT):
            weights_buffer.append(W3[i + j * core.N_OUTPUT])
    
    # Taking care of b3
    b3 = model.fc3.quant_bias().int().cpu().numpy().flatten()
    #print(max(b3))
    weights_buffer.extend(b3)
    
    # Convert float to bytes then to a list of int8
    scale_bytes = struct.pack('<f', scale)
    scale_int8_list = [int.from_bytes([byte], 'little', signed=True) for byte in scale_bytes]
    
    # Extend weights_buffer with scale_int8_list
    weights_buffer.extend(scale_int8_list)
    
    # Print the sent scale value
    print("Sent Scale:", scale)
    
    # Generate and send N_STEPS random numbers
    random_flag = False
    for i in range(core.N_STEPS):
        if ep_num <= core.START_EPISODE:
            random_number = np.arctanh(float(random.uniform(-1, 1))) # Generate a number that will be uniform after applying tanh
            random_flag = True
        else:
            random_number = float(random.gauss(0.0, 1.0))  # Generate a normally distributed random number
            random_flag = False

        random_numbers.append(random_number)
        random_bytes = struct.pack('<f', random_number)  # Convert float to bytes
        random_int8_list = [int.from_bytes([byte], 'little', signed=True) for byte in random_bytes]
        weights_buffer.extend(random_int8_list)
        
    model.train()
    if random_flag:
        print('Random episode')
    return weights_buffer, random_numbers