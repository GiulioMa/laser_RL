import gym
from gym import spaces
import numpy as np

class ContinuousRewardEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple environment where the reward is 1 if the observation is
    between 0.25 and 0.3, and 0 otherwise. The action is continuous.
    """
    def __init__(self):
        super(ContinuousRewardEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Initialize the state
        self.state = np.ones((1,)) * -2.0
        
    def step(self, action):
        # Apply action and add noise
        noise = np.random.normal(0, 0.01, size=(1,))
        self.state = self.state + action + noise
        
        # Ensure state is within the observation space bounds
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        
        # Define reward
        reward = 1.0 if 0.25 <= self.state[0] <= 0.3 else 0.0
        
        # Check if the episode is done
        done = False
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return self.state, reward, done, info
    
    def reset(self):
        # Reset the state 
        self.state = np.ones((1,)) * -2.0
        return self.state
    
    def render(self, mode='human'):
        # Render the environment to the screen
        print(f'Current state: {self.state[0]}')


class ActionNoiseEnv(gym.Env):
    """
    Custom Environment where the state is the action plus some noise.
    """
    def __init__(self):
        super(ActionNoiseEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Initialize the state
        self.state = np.zeros((1,))
        
    def step(self, action):
        # Add noise to the action
        noise = np.random.normal(0, 0.01, size=(1,))
        self.state = action + noise
        
        # Ensure state is within the observation space bounds
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        
        # Define reward
        reward = 1.0 if 0.25 <= self.state[0] <= 0.3 else 0.0
        
        # Check if the episode is done
        done = False
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return self.state, reward, done, info
    
    def reset(self):
        # Reset the state to zero
        self.state = np.zeros((1,))
        return self.state
    
    def render(self, mode='human'):
        # Render the environment to the screen
        print(f'Current state: {self.state[0]}')
        
        
class AbsDifferenceRewardEnv(gym.Env):
    """
    Custom Environment where the reward is the negative absolute difference
    between the state and 0.3. The action is continuous.
    """
    def __init__(self):
        super(AbsDifferenceRewardEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Initialize the state
        self.state = np.ones((1,)) * -2.0
        
    def step(self, action):
        # Apply action and add noise
        noise = np.random.normal(0, 0.01, size=(1,))
        self.state = self.state + action + noise
        
        # Ensure state is within the observation space bounds
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        
        # Define reward
        reward = -abs(self.state[0] - 0.3)
        
        # Check if the episode is done
        done = False
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return self.state, reward, done, info
    
    def reset(self):
        # Reset the state 
        self.state = np.ones((1,)) * -2.0
        return self.state
    
    def render(self, mode='human'):
        # Render the environment to the screen
        print(f'Current state: {self.state[0]}')


# Register environments
gym.envs.registration.register(
    id='ContinuousRewardEnv-v0',
    entry_point='__main__:ContinuousRewardEnv',
)

gym.envs.registration.register(
    id='ActionNoiseEnv-v0',
    entry_point='__main__:ActionNoiseEnv',
)

gym.envs.registration.register(
    id='AbsDifferenceRewardEnv-v0',
    entry_point='__main__:AbsDifferenceRewardEnv',
)

# Test the environments
if __name__ == '__main__':
    print("Testing ContinuousRewardEnv-v0")
    env = gym.make('ContinuousRewardEnv-v0')
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f'Action: {action}, Observation: {obs}, Reward: {reward}')
    
    print("\nTesting ActionNoiseEnv-v0")
    env = gym.make('ActionNoiseEnv-v0')
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        print(f'Action: {action}, Observation: {obs}, Reward: {reward}')
