from reinforcelearning import Policy, Quantizer
from torch.distributions import Categorical
import torch
import gym
import numpy as np

policy = Policy()
policy.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/policy_state_dict'))
policy.eval()


quantizer1 = Quantizer()
quantizer1.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/quantizer1_state_dict'))
quantizer1.eval()


quantizer2 = Quantizer()
quantizer2.load_state_dict(torch.load('/home/liang/PycharmProjects/SourceCoding/model/quantizer2_state_dict'))
quantizer2.eval()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def select_strategy1(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = quantizer1(state)
    m = Categorical(probs)
    action = m.sample()
    quantizer1.saved_log_probs.append(m.log_prob(action))
    return action.item()

def select_strategy2(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = quantizer2(state)
    m = Categorical(probs)
    action = m.sample()
    quantizer2.saved_log_probs.append(m.log_prob(action))
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
    bits1, bits2 = [],[]
    for t in range(150):

        env.render()
        stategy1 = select_strategy1(state)
        stategy2 = select_strategy2(state)
        # Channel
        state1 = uniform_midtread_quantizer(state, 1 / 2 ** stategy1)
        state2 = uniform_midtread_quantizer(state, 1 / 2 ** stategy2)
        bits1.append(stategy1)
        bits2.append(stategy2)
        state = np.stack([state1, state2]).flatten()
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print('Episode: ', i_episode, 'Last t: ', t, 'Bits used: ', sum(bits1))
    print('Episode: ', i_episode, 'Last t: ', t, 'Bits used: ', sum(bits2))
env.close()