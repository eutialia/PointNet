from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        # Each layer has batchnorm and relu on it
        # conv 3 64
        # conv 64 128
        # conv 128 1024
        # max pool
        # fc 1024 512
        # fc 512 256
        # fc 256 k*k (no batchnorm, no relu)
        # add bias
        # reshape
        self.conv_n3 = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv_n64 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.conv_n128 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.fc3 = nn.Linear(256, k*k)
    
    def forward(self, x):
        b, k, n = x.size()
        x = self.conv_n3(x)
        x = self.conv_n64(x)
        x = self.conv_n128(x)
        x = F.max_pool1d(x, kernel_size=n)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        bias = torch.eye(k).view(1, k*k).repeat(b, 1)
        if x.is_cuda:
            bias = bias.cuda()
        x = torch.add(x, bias).view((-1, k, k))
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        # conv 3 64
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        # conv 64 128
        # conv 128 1024 (no relu)
        # max pool
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.tnet1 = TNet(k=3)
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.tnet2 = TNet(k=64)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )

    def forward(self, x):
        n_pts = x.size()[2]

        # You will need these extra outputs:
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        trans = self.tnet1(x)
        x = torch.transpose(x, 1, 2)
        x = torch.bmm(x, trans)
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)

        if self.feature_transform:
            trans_feat = self.tnet2(x)
            x = torch.transpose(x, 1, 2)
            x = torch.bmm(x, trans_feat)
            x = torch.transpose(x, 1, 2)
        else:
            trans_feat = None

        pointfeat = x
        x = self.conv2(x)
        x = F.max_pool1d(x, n_pts)
        x = x.view(-1, 1024)
        if self.global_feat: # This shows if we're doing classification or segmentation
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        # get global features + point features from PointNetfeat
        # conv 1088 512
        # conv 512 256
        # conv 256 128
        # conv 128 k
        # softmax 
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.conv4 = nn.Conv1d(128, k, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        x, trans, trans_feat = self.feat(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.softmax(x)
        x = torch.transpose(x, 1, 2).contiguous()
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    # compute |((trans * trans.transpose) - I)|^2
    I = torch.eye(trans.size(-1)).repeat(trans.size(0), 1, 1)
    if trans.is_cuda:
        I = I.cuda()
    before_norm = torch.bmm(trans, torch.transpose(trans, 1, 2)) - I
    loss = torch.mean(torch.norm(before_norm, p=2, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 5)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
