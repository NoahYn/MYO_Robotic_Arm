import os
import torch
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
# import readchar
from video import record_video
from wrapper import CustomWrapper, NormalizeActionWrapper

from stable_baselines3 import A2C, PPO, SAC, DQN, TD3, HER, DDPG, TQC
# ARS A2C DDPG DQN HER PPO QR-DQN RecurrentPPO SAC TD3 TQC TRPO MaskablePPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

def make_env(env_id, rank, seed = 0) :
    def _init() :
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

env_id = "CartPole-v1"
PROCESSES_TO_TEST = [1, 2, 4, 8, 16]
NUM_EXPERIMENTS = 3
TRAIN_STEPS = 5000
EVAL_EPS = 20
ALGO = A2C

eval_env = gym.make(env_id)

reward_average = []
reward_std = []
training_times = []
total_procs = 0

for n_procs in PROCESSES_TO_TEST :
    total_procs += n_procs
    print(f"Testing {n_procs} processes")
    if n_procs == 1 :
        # only one process, no need to use multiprocessing
        train_env = DummyVecEnv([lambda : gym.make(env_id)])
    else :
        # using multiprocessing
        train_env = SubprocVecEnv(
            [make_env(env_id, i + total_procs) for i in range(n_procs)],
            start_method="fork",
        )
    rewards = []
    times = []
    
    for experiment in range(NUM_EXPERIMENTS) :
        train_env.reset()
        model = ALGO("MlpPolicy", train_env, verbose=0)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        times.append(time.time() - start)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
    # important : don't forget to close subprocesses
    train_env.close()
    reward_average.append(np.mean(rewards))
    reward_std.append(np.mean(rewards))
    training_times.append(np.mean(times))

def plot_training_resultss(training_steps_per_second, reward_averages, reward_std) :
    """
    Utility function for plotting the resurts of training.

    Args:
        training_steps_per_second : List[double]
        reward_averages : List[double]
        reward_std : List[double]
    """
    plt.figure(figsize = (9, 4))  
    plt.subplots_adjust(wspace = 0.5)
    plt.subplot(1, 2, 1)
    # plt.errorbar(
    #     PROCESSES_TO_TEST,
    #     reward_averages,
    #     yerr = reward_std,
    #     capsize = 2,
    #     c = 'k'
    #     marker = 'o',
    # )
    plt.xlabel("Processes")
    plt.ylabel("Average return")
    plt.subplot(1, 2, 2)
    plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
    plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
    plt.xlabel("Processes")
    plt.ylabel("Training steps per second")

training_steps_per_second = [TRAIN_STEPS / t for t in training_times]
plot_training_resultss(training_steps_per_second, reward_average, reward_std)    



env = gym.make(env_id, render_mode = "rgb_array")
env = NormalizeActionWrapper(env)

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