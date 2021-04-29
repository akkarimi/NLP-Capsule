import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def squash_v1(x, axis):
    s_squared_norm = (x ** 2).sum(axis, keepdim=True)
    scale = torch.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)
        torch.nn.init.xavier_uniform_(self.capsules.weight)
        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x):
        batch_size = x.size(0)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        poses = squash_v1(u, axis=1)
        activations = torch.sqrt((poses ** 2).sum(1))
        # x.shape: torch.Size([16, 32, 1148])
        # self.capsules(x).shape: torch.Size([16, 256, 1148])
        # u.shape: torch.Size([16, 8, 32, 1148, 1])
        # poses.shape: torch.Size([16, 8, 32, 1148, 1])
        # activations.shape: torch.Size([16, 32, 1148, 1])
        return poses, activations

class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()
    def forward(self, p, a):
        poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)
        activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)
        return poses, activations


class FCCaps(nn.Module):
    def __init__(self, args, output_capsule_num, input_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num
        self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, out_channels, in_channels))
        torch.nn.init.xavier_uniform_(self.W1)

    def forward(self, x, y):
        batch_size = x.size(0)
        x = torch.stack([x] * self.output_capsule_num, dim=2).unsqueeze(4)
        W1 = self.W1.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W1, x)
        b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, self.output_capsule_num, 1)).cuda()
        # b_ij: [16, 128, 3, 1]
        # u_hat: [16, 128, 3, 8, 1]
        num_iterations = 10
        for i in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2).unsqueeze(4) # [16, 128, 3, 1, 1]
            s_ij = (c_ij * u_hat).sum(dim=1, keepdim=True) # [16, 1, 3, 8, 1]
            v_j = squash_v1(s_ij, axis=1) # [16, 1, 3, 8, 1]
            v_j_i = torch.cat([v_j] * self.input_capsule_num, dim=1)
            v_j_i = (v_j_i * u_hat).sum(3)
            b_ij = b_ij + v_j_i # b_ij: [16, 128, 3, 1]
        poses = v_j.squeeze(1)
        activations = torch.sqrt((poses ** 2).sum(2)).squeeze(-1)
        return poses, activations

