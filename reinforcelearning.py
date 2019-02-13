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
import os
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
        self.affine1 = nn.Linear(8, 256)
        self.affine2 = nn.Linear(256, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)



class Quantizer(nn.Module):

    def __init__(self):
        super(Quantizer, self).__init__()
        self.affine1 = nn.Linear(4, 256)
        self.affine2 = nn.Linear(256, 8)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)



policy = Policy()
quantizer1 = Quantizer()
quantizer2 = Quantizer()

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
optimizer_q1 = optim.Adam(quantizer1.parameters(), lr=1e-2)
optimizer_q2 = optim.Adam(quantizer2.parameters(), lr=1e-2)

eps = np.finfo(np.float32).eps.item()


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



def quantizer1_finish_episode():
    R = 0
    quantizer_loss = []
    rewards = []
    for r in quantizer1.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(quantizer1.saved_log_probs, rewards):
        quantizer_loss.append(-log_prob * reward)
    optimizer_q1.zero_grad()
    quantizer_loss = torch.cat(quantizer_loss).sum()
    quantizer_loss.backward()
    optimizer_q1.step()
    del quantizer1.rewards[:]
    del quantizer1.saved_log_probs[:]

def quantizer2_finish_episode():
    R = 0
    quantizer_loss = []
    rewards = []
    for r in quantizer2.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(quantizer2.saved_log_probs, rewards):
        quantizer_loss.append(-log_prob * reward)
    optimizer_q1.zero_grad()
    quantizer_loss = torch.cat(quantizer_loss).sum()
    quantizer_loss.backward()
    optimizer_q1.step()
    del quantizer2.rewards[:]
    del quantizer2.saved_log_probs[:]


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
    if len(durations_t) >= 5:
        means = durations_t.unfold(0, 5, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(4), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

episode_strategies = []

def plot_strategies():
    plt.figure(3)
    plt.clf()
    durations_s = torch.tensor(episode_strategies, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Strategies')
    plt.plot(durations_s.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_s) >= 5:
        means = durations_s.unfold(0, 5, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(4), means))
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
    c1,c2 = [1,1,1,1],[1,1,1,1]
    for i_episode in count(1):
        state = env.reset()
        feedback1 = np.zeros(state.shape)
        feedback2 = np.zeros(state.shape)
        for t in range(10000):  # Don't infinite loop while learning
            # env.render()
            # Begin source coding here
            # encoder:
            encoder_state1 = c1*state - feedback1
            encoder_state2 = c2*state - feedback2
            stategy1 = select_strategy1(encoder_state1) # learn f_encoder
            quantized_state1 = uniform_midtread_quantizer(encoder_state1, 1 / 2 ** stategy1)
            stategy2 = select_strategy2(encoder_state2)  # learn f_encoder
            quantized_state2 = uniform_midtread_quantizer(encoder_state2, 1 / 2 ** stategy2)
            # feedback:
            feedback1 = quantized_state1 + feedback1
            feedback2 = quantized_state2 + feedback2
            # print(feedback)
            control_state = np.stack([feedback1, feedback2]).flatten()
            control_state = np.stack([state, state]).flatten()

            # Controller:
            action = select_action(control_state) # learn f_decoder
            state, reward, done, _ = env.step(action)
            # reward = reward -(abs(state[2]))
            # if done:
            #     print(reward)
            # finish source coding here
            if args.render:
                env.render()
            rr = reward #- stategy1*0.1-stategy2*0.1
            policy.rewards.append(rr)
            quantizer1.rewards.append(rr)
            quantizer2.rewards.append(rr)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01


        finish_episode()
        quantizer1_finish_episode()
        quantizer2_finish_episode()

        episode_durations.append(t + 1)
        episode_strategies.append(stategy1+stategy2)
        plot_strategies()
        plot_durations() # Plot the durations
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold: #(195)
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        if i_episode % 50 == 0:
            folder_path = os.path.join('/home/liang/PycharmProjects/SourceCoding/', 'model')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            torch.save(policy.state_dict(), folder_path + '/policy_state_dict')
            torch.save(quantizer1.state_dict(), folder_path+'/quantizer1_state_dict')
            torch.save(quantizer2.state_dict(), folder_path + '/quantizer2_state_dict')

    torch.save(policy.state_dict(), folder_path + '/policy_state_dict')
    torch.save(quantizer1.state_dict(), folder_path + '/quantizer1_state_dict')
    torch.save(quantizer2.state_dict(), folder_path + '/quantizer2_state_dict')
    print('Complete')
    import pickle
    duration = open('duration2.pickle', 'wb')
    pickle.dump(episode_durations, duration)
    duration.close()
    strategies = open('strategies2.pickle','wb')
    pickle.dump(episode_strategies, strategies)
    strategies.close()
    env.close()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()