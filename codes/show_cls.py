from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
dataset_root = '../shapenet'
opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root=dataset_root,
    classification=True,
    split='test',
    npoints=opt.num_points)

train_dataset = ShapeNetDataset(
    root=dataset_root,
    classification=True,
    npoints=opt.num_points)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8)

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

total_train = 0
correct_train = 0

total_test = 0
correct_test = 0
with torch.no_grad():
    for i, data in enumerate(train_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans, trans_feat = classifier(points)
        pred_choice = pred.data.max(1)[1]
        total_train += target.size(0)
        correct_train += pred_choice.eq(target.data).cpu().sum().item()
    print(f'Train accuracy: {100 * correct_train / total_train}')

    for i, data in enumerate(test_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans, trans_feat = classifier(points)
        pred_choice = pred.data.max(1)[1]
        total_test += target.size(0)
        correct_test += pred_choice.eq(target.data).cpu().sum().item()
    print(f'Test accuracy: {100 * correct_test / total_test}')