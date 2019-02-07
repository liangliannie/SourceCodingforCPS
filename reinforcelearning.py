import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pylab as plt
import matplotlib
from PIL import Image
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)



class Quantizer(nn.Module):

    def __init__(self):
        super(Quantizer, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 8)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)



policy = Policy()
quantizer = Quantizer()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
optimizer_q = optim.Adam(policy.parameters(), lr=1e-2)

eps = np.finfo(np.float32).eps.item()


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

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]



def quantizer_finish_episode():
    R = 0
    quantizer_loss = []
    rewards = []
    for r in quantizer.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(quantizer.saved_log_probs, rewards):
        quantizer_loss.append(-log_prob * reward)
    optimizer_q.zero_grad()
    quantizer_loss = torch.cat(quantizer_loss).sum()
    quantizer_loss.backward()
    optimizer_q.step()
    del quantizer.rewards[:]
    del quantizer.saved_log_probs[:]


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def uniform_midtread_quantizer(x, Q):
    # limiter
    x = np.copy(x)
    idx = np.where(np.abs(x) >= 1)
    x[idx] = np.sign(x[idx])
    # linear uniform quantization
    xQ = Q * np.floor(x/Q + 1/2)
    return xQ

def main():
    plt.ion()
    running_reward = 10
    env.reset()
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            # env.render()
            action = select_action(state)
            stategy = select_strategy(state)
            state, reward, done, _ = env.step(action)
            # source coding here
            # What is the feedback value?
            state = uniform_midtread_quantizer(state, 1/2**stategy)
            # finish souce coding here
            if args.render:
                env.render()
            policy.rewards.append(reward)
            quantizer.rewards.append(-stategy)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01


        finish_episode()
        quantizer_finish_episode()
        episode_durations.append(t + 1)

        plot_durations() # Plot the durations
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold: #(195)
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

    print('Complete')
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()