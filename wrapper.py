import gymnasium as gym
import numpy as np

class CustomWrapper(gym.Wrapper) :
    """
    param env : (gym.Env) environment to wrap
    """
    def __init__(self, env) :
        super().__init__(env)
        
    def reset(self, **kwargs) :     
        return self.env.reset(**kwargs)
    
    def step(self, action) :
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return obs, reward, terminated, truncated, info
    
class NormalizeActionWrapper(gym.Wrapper) :
    def __init__(self, env) :
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only wokrs with continuous action space (spaces.Box)"
        
        # Retrive the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(
            low = -1, high = 1, shape = action_space.shape, dtype = np.float32
        )
        
        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action) :
        """
        Rescale the action from [-1, 1] to [low, high]
        param scaled_action : (np.ndarray)
        return : (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))
    
    def reset(self, **kwargs) : 
        return self.env.reset(**kwargs)
    
    def step(self, action) :
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info
    
class MonitorWrapper(gym.Wrapper) :
    
    def __init__(self, env) :
        super().__init__(env)
        self.env = env
        self.epi_len = 0
        self.epi_ret = 0
        
    def reset(self, **kwargs) :
        obs = self.env.reset(**kwargs)
        self.epi_len = 0
        self.epi_ret = 0
        return obs
    
    def step(self, action) :
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.epi_len += 1
        self.epi_ret += reward
        if terminated :
            info['episode'] = {'r': self.epi_ret, 'l': self.epi_len}
        return obs, reward, terminated, truncated, info