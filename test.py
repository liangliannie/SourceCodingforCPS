from reinforcelearning import Policy, Quantizer
from torch.distributions import Categorical
import torch
import gym
import numpy as np

policy = Policy()
policy.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/policy_state_dict'))
policy.eval()


quantizer = Quantizer()
quantizer.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/quantizer_state_dict'))
quantizer.eval()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def select_strategy(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = quantizer(state)
    m = Categorical(probs)
    action = m.sample()
    quantizer.saved_log_probs.append(m.log_prob(action))
    return action.item()


def uniform_midtread_quantizer(x, Q):
    # limiter
    x = np.copy(x)
    idx = np.where(np.abs(x) >= 1)
    x[idx] = np.sign(x[idx])
    # linear uniform quantization
    xQ = Q * np.floor(x/Q + 1/2)
    return xQ


env = gym.make('CartPole-v0')
for i_episode in range(20):
    state = env.reset()
    bits = []
    for t in range(150):

        env.render()
        stategy = select_strategy(state)
        # Channel
        state = uniform_midtread_quantizer(state, 1 / 2 ** stategy)
        bits.append(stategy)
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print('Episode: ', i_episode, 'Last t: ', t, 'Bits used: ', sum(bits))
env.close()