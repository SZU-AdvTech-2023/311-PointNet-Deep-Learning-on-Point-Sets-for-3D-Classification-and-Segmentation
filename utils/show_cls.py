from __future__ import print_function
import argparse

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F


def iou_mean(pred, target, n_classes=1):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float(0))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes, ious


# showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='D:\\Project\\pointnet.pytorch-master\\utils\\cls1\\cls_model_1.pth',
                    help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root='D:\\Project\\pointnet.pytorch-master\\data\\shapenetcore_partanno_segmentation_benchmark_v0',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True)

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()
classes_num = test_dataset.classes.__len__()
mious = 0
ious = np.zeros(classes_num)
acc = 0

for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, _, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    mious_i, ious_i = iou_mean(pred_choice, target, classes_num)
    mious += mious_i
    ious += np.array(ious_i)
    acc += correct / float(32)
    print('i:%d  loss: %f accuracy: %f mious: %f' % (i, loss.data.item(), correct / float(32), mious_i))

mious /= i
ious /= i
acc /= i
print('final miou: %f' % (mious))