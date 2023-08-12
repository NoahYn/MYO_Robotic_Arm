import os
import torch
import gymnasium as gym
from video import record_video
from stable_baselines3 import A2C, PPO, SAC, DQN, TD3, HER, DDPG, TQC
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv   

print(torch.cuda.is_available())
# import readchar

env = gym.make("CartPole-v1", render_mode = "rgb_array")

model = A2C("MlpPolicy", env, verbose=1).learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000) :
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    
    # if done :
    #     obs = env.reset()
    #     break
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
record_video("CartPole-v1", model, video_length=500, prefix="ppo-cartpole")
os.makedirs("/saved_model", exist_ok=True)
model.save("/saved_model/ppo_cartpole")

env.close()