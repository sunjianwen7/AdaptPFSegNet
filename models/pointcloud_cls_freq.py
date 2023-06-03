import open3d
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetFreqExtract

# 0.903722
#  0.923780

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel
        self.freE = PointNetFreqExtract(large_scale=0.2,small_scale=0.1,mlp=[64,128,256,512,1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
    def forward(self, xyz):
        B, _, _ = xyz.shape
        xyz = xyz[:, :3, :]

        points=self.freE(xyz)

        x = points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x,points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
