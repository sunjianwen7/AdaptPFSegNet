"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os

import open3d

from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np




class show_partseg:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37],
                   'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                   'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    @staticmethod
    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

    def __init__(self,num_votes):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        experiment_dir = '/home/sunjianwen/Pointnet_Pointnet2_pytorch/log/part_seg/test'
        root = '/home/sunjianwen/Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        TEST_DATASET = PartNormalDataset(root=root, npoints=2048, split='test', normal_channel=False)
        self.testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)
        self.num_classes = 16
        self.num_part = 50
        self.num_votes=num_votes
        '''MODEL LOADING'''
        model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
        MODEL = importlib.import_module(model_name)
        self.classifier = MODEL.get_model(self.num_part).cuda()
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat
        with torch.no_grad():

            self.data_list = []
            for batch_id, (points, label, target) in tqdm(enumerate(self.testDataLoader), total=len(self.testDataLoader),
                                                          smoothing=0.9):
                self.data_list.append([points, label, target])
    def get_index(self,index):
        data = self.data_list[index]
        points = data[0]
        batchsize, num_point, _ = points.size()
        pcd = open3d.geometry.PointCloud()
        points=points[0].detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(points)
        color = np.array([0.5, 0.5, 1.0])  # 浅蓝色
        colors=np.array([color for i in range(len(points))])
        pcd.colors = open3d.utility.Vector3dVector(colors)
        return pcd
    def predict(self,index):
        '''HYPER PARAMETER'''
        with torch.no_grad():
            classifier = self.classifier.eval()
            data=self.data_list[index]
            points=data[0]
            label = data[1]
            target = data[2]
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], self.num_part).cuda()
            for _ in range(self.num_votes):
                seg_pred, _ = classifier(points, self.to_categorical(label, self.num_classes))
                vote_pool += seg_pred
            seg_pred = vote_pool / self.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            point_colors=[]
            for i in range(cur_batch_size):
                cat = self.seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, self.seg_classes[cat]], 1) + self.seg_classes[cat][0]
            for i, intensity in enumerate(cur_pred_val[0]):
                point_colors.append([intensity / 3, 0, 1 - intensity / 3])
            colos = open3d.utility.Vector3dVector(point_colors)
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points.permute(0, 2, 1)[0].detach().cpu().numpy())
            pcd.colors = colos

            return pcd
    def display_pc(self,pcd):
        open3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    test=show_partseg(3)
    # test.fill_data()
    pcd=test.get_index(1)
    test.display_pc(pcd)
