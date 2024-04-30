import torch
from torch import nn
import copy
import errno
import os
from torch.autograd import Variable


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class DeepQNetwork(nn.Module):

    def __init__(self, num_actions):

        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=8, padding=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=1),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(256, 512, bias=True),
            nn.ReLU()
        ) 
        self.out = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )
        if torch.cuda.is_available():
            self.cuda()
        self.apply(self.weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.out(x)
        return x

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        if classname.find('Linear') != -1:
            pass


def Q_targets(phi_plus1_mb, r_mb, done_mb, model, gamma=0.99):
    '''
    gamma: future reward discount factor
    '''
    x = torch.from_numpy(phi_plus1_mb).float()
    max_Q, argmax_a = model(x).max(1)
    max_Q = max_Q.detach()
    target = torch.from_numpy(r_mb).float() + (gamma * max_Q) * \
        (1 - torch.from_numpy(done_mb).float())
    target = target.unsqueeze(1)
    return target


def Q_values(model, phi_mb, action_mb):
    q_phi = model(torch.from_numpy(phi_mb).float())
    action_mb = torch.from_numpy(action_mb).long().unsqueeze(1)
    q_phi = q_phi.gather(1, action_mb)
    return q_phi


def gradient_descent(y, q, optimizer):
    optimizer.zero_grad()

    error = (y - q)

    error = error**2
    error = error.sum()
    error.backward()

    optimizer.step()

    return q, error


def copy_network(Q):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2

def save_network(model, episode, out_dir):
    out_dir = '{}/models'.format(out_dir)
    make_dir(out_dir)
    torch.save(model.state_dict(), '{}/episode_{}'.format(out_dir, episode))
