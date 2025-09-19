# ============================
# FILE: gomoku/policy_value_net.py
# ============================
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super().__init__()
        c_in = 4
        self.board_width = board_width
        self.board_height = board_height
        self.conv1 = nn.Conv2d(c_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # policy 分支
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # value 分支
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # policy
        p = F.relu(self.act_conv1(x))
        p = p.view(-1, 4 * self.board_width * self.board_height)
        p = F.log_softmax(self.act_fc1(p), dim=1)
        # value
        v = F.relu(self.val_conv1(x))
        v = v.view(-1, 2 * self.board_width * self.board_height)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v))
        return p, v


class PolicyValueNet:
    def __init__(self, board_width, board_height, model_file=None, l2_const=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(board_width, board_height).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=l2_const)
        if model_file and os.path.exists(model_file):
            state = torch.load(model_file, map_location=self.device)
            self.net.load_state_dict(state)


    def policy_value(self, state_batch):
        s = Variable(torch.FloatTensor(state_batch).to(self.device))
        log_p, v = self.net(s)
        p = torch.exp(log_p).detach().cpu().numpy()
        return p, v.detach().cpu().numpy()


    def policy_value_fn(self, board_env):
        legal = board_env.availables
        s = np.ascontiguousarray(board_env.current_state().reshape(-1, 4, self.net.board_width, self.net.board_height))
        log_p, v = self.net(Variable(torch.from_numpy(s)).to(self.device).float())
        p = torch.exp(log_p).detach().cpu().numpy().flatten()
        return list(zip(legal, p[legal])), v.detach().cpu().numpy()[0][0]


    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        s = Variable(torch.FloatTensor(state_batch).to(self.device))
        m = Variable(torch.FloatTensor(mcts_probs).to(self.device))
        w = Variable(torch.FloatTensor(winner_batch).to(self.device))
        # 设置学习率
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.zero_grad()
        log_p, v = self.net(s)
        value_loss = F.mse_loss(v.view(-1), w)
        policy_loss = -torch.mean(torch.sum(m * log_p, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        # 熵（仅监控）
        entropy = -torch.mean(torch.sum(torch.exp(log_p) * log_p, dim=1)).item()
        return loss.item(), entropy


    def save_model(self, path):
        torch.save(self.net.state_dict(), path)