import gym
import time
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import pdb

from dqn_agent import Agent
agent = Agent(state_size=8, action_size=4 , seed=0)

env = gym.make('LunarLander-v2')


eps = 1
scores = []
scores_window = deque(maxlen=100)  # last 100 scores
# env is created, now we can use it:
pdb.set_trace()
for episode in range(10000):
    observation = env.reset()
    agent.reset_state(observation)
    done = False
    score = 0
    t_start = time.time()
    while not done:
        action = agent.act(observation, eps)
        next_observation, reward, done, info = env.step(action)
        # observations are (210, 160, 3) RGB images
        agent.step(observation, action, reward, next_observation, done)
        observation = next_observation
        score += reward
        env.render()
    t_end = time.time()
    scores_window.append(score)
    scores.append(score)
    if episode %10 == 0:
        agent.qnetwork_local.save()  # saves the model weights after training
    print("\rEpisode {}\tAverage Score: {:.2f}\tElapsed Time:{} (s)".format(episode, np.mean(scores_window), t_end-t_start))
    eps = max(0.01, 0.995*eps)  # decay epsilon to a minimum of 0.01
agent.qnetwork_local.save()  # saves the model weights after training
env.close()