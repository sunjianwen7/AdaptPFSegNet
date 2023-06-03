import open3d
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.SegNet_utils import LayeredExtractMoudle_part, UNsamplingMoudle
from train_semseg import writer
import torchvision.utils as vutils

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.lem1 = LayeredExtractMoudle_part(mlps=[3,32, 64, 256],npoint=512,density_list=[1,2],sample_num=32,radiu=0.1, small=0.1, large=0.2,block_num=2)
        self.lem2 = LayeredExtractMoudle_part(mlps=[256, 64, 128, 512],npoint=128, density_list=[1,2], sample_num=32, radiu=0.2, small=0.2, large=0.4,block_num=2)
        self.lem3 = LayeredExtractMoudle_part(mlps=[512, 256, 512, 1024],npoint=None, density_list=None, sample_num=None, radiu=None, small=None, large=None,block_num=None,all_extra=True)
        self.usm1 = UNsamplingMoudle(mlps=[1536,256,256])
        self.usm2 = UNsamplingMoudle(mlps=[512, 256, 128])
        self.usm3 = UNsamplingMoudle(mlps=[150, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    def forward(self, xyz, cls_label):
        xyz = xyz.permute(0, 2, 1)[:,:,:3]
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(xyz[0].cpu().numpy())
        # pcd.paint_uniform_color([0.6, 0.8, 1.0])
        # open3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
        #                                      window_name="点云显示",
        #                                      point_show_normal=False,
        #                                      width=800,  # 窗口宽度
        #                                      height=600)
        feature=xyz
        B,N,C = xyz.shape
        l1_xyz, l1_feature = self.lem1(xyz,feature)
        writer.add_image('features1', vutils.make_grid(l1_feature[0].cpu().unsqueeze(dim=0), normalize=True, scale_each=True), global_step=0)
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(l1_xyz[0].cpu().numpy())
        # pcd.paint_uniform_color([0.6, 0.8, 1.0])
        # open3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
        #                                   window_name="点云显示",
        #                                   point_show_normal=False,
        #                                   width=800,  # 窗口宽度
        #                                   height=600)
        l2_xyz, l2_feature = self.lem2(l1_xyz, l1_feature)
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(l2_xyz[0].cpu().numpy())
        # pcd.paint_uniform_color([0.6, 0.8, 1.0])
        # open3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
        #                                   window_name="点云显示",
        #                                   point_show_normal=False,
        #                                   width=800,  # 窗口宽度
        #                                   height=600)
        writer.add_image('features2',
                         vutils.make_grid(l2_feature[0].cpu().unsqueeze(dim=0), normalize=True, scale_each=True),
                         global_step=0)
        l3_xyz, l3_feature = self.lem3(l2_xyz, l2_feature)
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(l3_xyz[0].cpu().numpy())
        # pcd.paint_uniform_color([0.6, 0.8, 1.0])
        # open3d.visualization.draw_geometries([pcd],  # 待显示的点云列表
        #                                   window_name="点云显示",
        #                                   point_show_normal=False,
        #                                   width=800,  # 窗口宽度
        #                                   height=600)
        writer.add_image('features3',
                         vutils.make_grid(l3_feature[0].cpu().unsqueeze(dim=0), normalize=True, scale_each=True),
                         global_step=0)
        # Feature Propagation layers
        new_l2_feature = self.usm1(l2_xyz,l3_xyz, l2_feature, l3_feature)
        writer.add_image('features4',
                         vutils.make_grid(new_l2_feature[0].cpu().unsqueeze(dim=0), normalize=True, scale_each=True),
                         global_step=0)
        new_l1_feature = self.usm2(l1_xyz,l2_xyz, l1_feature, new_l2_feature)
        writer.add_image('features5',
                         vutils.make_grid(new_l1_feature[0].cpu().unsqueeze(dim=0), normalize=True, scale_each=True),
                         global_step=0)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N).permute(0, 2, 1)
        new_xyz = self.usm3( xyz,l1_xyz,torch.cat([cls_label_one_hot,xyz,feature],2), new_l1_feature)
        writer.add_image('features6',
                         vutils.make_grid(new_xyz[0].cpu().unsqueeze(dim=0), normalize=True, scale_each=True),
                         global_step=0)
        # FC layers
        x = new_xyz.permute(0, 2, 1)
        feat = F.relu(self.bn1(self.conv1(x )))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_feature


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss