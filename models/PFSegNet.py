import torch.nn as nn
import torch
import torch.nn.functional as F

from models.SegNet_utils import LayeredExtractMoudle_seg,UNsamplingMoudle
from train_semseg import writer


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.lem1 = LayeredExtractMoudle_seg(mlps=[3,32, 32, 64],npoint=1024,density_list=[1,2],sample_num=32,radiu=0.1, small=0.1, large=0.5,block_num=2)
        self.lem2 = LayeredExtractMoudle_seg(mlps=[64, 64, 64, 128],npoint=256, density_list=[1,2], sample_num=32, radiu=0.2, small=0.2, large=1.0,block_num=2)
        self.lem3 = LayeredExtractMoudle_seg(mlps=[128, 128, 256, 512],npoint=16, density_list=None, sample_num=32, radiu=0.8, small=None, large=None,block_num=None,all_extra=True)
        self.usm1 = UNsamplingMoudle(mlps=[640,256,256])
        self.usm2 = UNsamplingMoudle(mlps=[320, 256, 256])
        self.usm3 = UNsamplingMoudle(mlps=[259, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        # print('SegNet input xyz', xyz.shape )
        xyz = xyz.permute(0, 2, 1)[:,:,:3]
        feature=xyz

        l1_xyz, l1_feature = self.lem1(xyz,feature)
        l2_xyz, l2_feature = self.lem2(l1_xyz, l1_feature)
        l3_xyz, l3_feature = self.lem3(l2_xyz, l2_feature)
        # Feature Propagation layers
        new_l2_feature = self.usm1(l2_xyz,l3_xyz, l2_feature, l3_feature)
        new_l1_feature = self.usm2(l1_xyz,l2_xyz, l1_feature, new_l2_feature)
        new_xyz = self.usm3( xyz,l1_xyz,feature, new_l1_feature)
        new_xyz = new_xyz.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(new_xyz))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_feature
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        regularizer_weight=1
        CrossEntropy_Loss = F.nll_loss(pred, target, weight=weight)
        K = trans_feat.size(2)
        device = trans_feat.device
        product = torch.matmul(trans_feat.transpose(1, 2), trans_feat)
        mat_diff =torch.eye(K, dtype=torch.float32,device=device) - product
        mat_diff_loss = torch.mean(torch.norm(mat_diff, p='fro'))
        total_loss=regularizer_weight * CrossEntropy_Loss+mat_diff_loss* (1 - regularizer_weight)
        return total_loss,[CrossEntropy_Loss,mat_diff_loss]