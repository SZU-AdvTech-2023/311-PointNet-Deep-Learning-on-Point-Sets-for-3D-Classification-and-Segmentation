from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F
import open3d
import numpy as np
import nearest_neighbors

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='D:\\Project\\pointnet.pytorch-master\\utils\\cls\\cls_model_1.pth',
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

pc = []
labelToColor = dict()

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

# 赋予颜色
for i in range(len(test_dataset.classes)):
    x = (i * 0.25) % 1
    y = (((i * 0.25) // 1) * 0.25) % 1
    z = (((((i * 0.25) // 1) * 0.25) // 1) * 0.25) % 1
    labelToColor[i] = [x, y, z]

for i, data in enumerate(testdataloader, 0):
    points, target = data

    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    for k in range(32):
        offset_pos = points[0, 0:1, :] + 2*k
        points[k, :, :] = offset_pos
        target[k] = target[0]
    pred, _, _ = classifier(points)

    pred_choice = pred.data.max(1)[1].cpu().numpy()
    for j in range(points.shape[0]):
        point_cloud = open3d.geometry.PointCloud()
        label_point_cloud = open3d.geometry.PointCloud()
        point_set = points.cpu().numpy()[j].transpose()
        point_set[:, 0] += 2 * j
        point_cloud.points = open3d.utility.Vector3dVector(point_set)
        point_cloud.colors = open3d.utility.Vector3dVector(
            np.ones(point_set.shape) * labelToColor[np.squeeze(target.cpu().numpy())[j]])
        point_set[:, 1] += 2
        label_point_cloud.points = open3d.utility.Vector3dVector(point_set)
        label_point_cloud.colors = open3d.utility.Vector3dVector(
            np.ones(point_set.shape) * labelToColor[pred_choice[j]])
        pc.append(point_cloud)
        pc.append(label_point_cloud)

open3d.visualization.draw_geometries(pc)
