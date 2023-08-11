import gymnasium as gym
import readchar

import random  

env = gym.make("MountainCar-v0", render_mode = "human")
# env = gym.make("CartPole-v1", render_mode = "human")
observation, info = env.reset(seed=42)

while True :
    env.render()
    action = int(readchar.readkey())

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated :
        observation, info = env.reset()
        break
        
env.close()