import numpy as np
import open3d
import torch
from torch import nn
import torch.nn.functional as F
import time
import torchvision.utils as vutils
from train_semseg import writer


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f" 函数名: {func.__name__}, 运行时间: {execution_time} 秒")
        return result
    return wrapper
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
def calculate_point_cloud_volume(points):
    # 计算点云的边界框
    min_coords = torch.min(points, dim=1).values
    max_coords = torch.max(points, dim=1).values

    # 计算边界框的尺寸
    dimensions = max_coords - min_coords
    # 计算体积
    volume = torch.prod(dimensions, dim=1, keepdim=True)
    return volume

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # print("farthest_point_sample xyz",xyz.shape)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def get_normals(radius, nsample, xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
    Return:
        group_idx: grouped points index, [B, N, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, N, 1])
    sqrdists = square_distance(xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, N, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    # print(mask)
    # print(group_idx.shape)
    # print(group_first.shape)
    group_idx[mask] = group_first[mask]
    group_points=index_points(xyz,group_idx)
    center = group_points.mean(dim=2, keepdim=True)  # [B, N, 1, 3]

    # 中心化点云
    centered_points = group_points - center  # [B, N, S, 3]

    # 计算协方差矩阵
    cov_matrix = torch.matmul(centered_points.transpose(2, 3), centered_points)  # [B, N, 3, 3]

    # 计算特征向量
    _, eigen_vectors = torch.linalg.eigh(cov_matrix)  # [B, N, 3]

    # 提取法向量（最小特征值对应的特征向量）
    normals = eigen_vectors[..., 0]  # [B, N, 3]

    return normals

def get_normal(points,radius):
    # 计算每个点的法向量
    normals = []
    for i, point in enumerate(points):
        dists = np.linalg.norm(points - point, axis=1)
        sorted_indices = np.argsort(dists)
        indices = sorted_indices[dists[sorted_indices] <= radius]
        if len(indices) < 3:  # 确保有足够多的邻居点用于计算法向量
            normals.append([0, 0, 0])
            continue
        centroid = np.mean(points[indices], axis=0)
        cov = np.dot((points[indices] - centroid).T, (points[indices] - centroid))
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argsort(eigvals)[::-1]
        normal = eigvecs[:, idx[-1]]
        normals.append(normal)
    normals = np.asarray(normals)
    return normals
# @calculate_time
def get_curvate(pointcloud:torch.tensor,large_scale:float,small_scale:float)->torch.tensor:
    points=pointcloud[:,:, :3]
    small=get_normals(small_scale, 32, points)
    large=get_normals(large_scale,64,points)
    diff_normals = (small - large) / 2
    # 计算模长
    curvature = torch.linalg.norm(diff_normals, axis=2)

    return curvature


class PointExtractMoudle(nn.Module):
    def __init__(self, mlps):
        super(PointExtractMoudle, self).__init__()
        self.conv1 = torch.nn.Conv1d(mlps[0],mlps[1], 1)
        self.conv2 = torch.nn.Conv1d(mlps[1], mlps[2], 1)
        self.conv3 = torch.nn.Conv1d(mlps[2], mlps[3], 1)
        self.bn1 = nn.BatchNorm1d(mlps[1])
        self.bn2 = nn.BatchNorm1d(mlps[2])
        self.bn3 = nn.BatchNorm1d(mlps[3])

    def forward(self, x):
        pointfeat = x[:, :, :3]
        pointfeat = F.relu(self.bn1(self.conv1(pointfeat)))
        pointfeat = F.relu(self.bn2(self.conv2(pointfeat)))
        pointfeat = self.bn3(self.conv3(pointfeat))
        x = torch.cat([x, pointfeat], dim=2)
        return x
class FreqExtractMoudle(nn.Module):
    def __init__(self):
        super(FreqExtractMoudle, self).__init__()

        self.conv1 = torch.nn.Conv1d(1,16, 1)
        self.conv2 = torch.nn.Conv1d(16, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 16, 1)
        self.conv4 = torch.nn.Conv1d(16, 1, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(1)

    # @calculate_time
    def forward(self, x,small,large):
        pointxyz = x[:, :, :3]
        # print("FreqExtractMoudle pointxyz:",pointxyz.shape)
        curvates = get_curvate(pointxyz, large , small).unsqueeze(-1)
        # curvates = torch.from_numpy(curvates).float().to('cuda')
        curvates = curvates.permute(0, 2, 1)
        # print("FreqExtractMoudle curvates:", curvates.shape)
        curvates = F.relu(self.bn1(self.conv1(curvates)))
        curvates = F.relu(self.bn2(self.conv2(curvates)))
        curvates = F.relu(self.bn3(self.conv3(curvates)))
        curvates = self.bn4(self.conv4(curvates))
        curvates = torch.squeeze(curvates, dim=1)
        return curvates



class SampleGroupMoudle_part(nn.Module):


    def __init__(self,key_num,):
        super(SampleGroupMoudle_part, self).__init__()
        self.key_num=int(key_num)

    # @calculate_time
    def forward(self,x):
        x=x[:, :, :3]
        fps_idx = farthest_point_sample(x,self.key_num)  # [B, keypoint_num, D]
        return fps_idx





class SampleGroupMoudle(nn.Module):
    def __init__(self,density,):
        super(SampleGroupMoudle, self).__init__()
        self.density=density

    # @calculate_time
    def forward(self,x):
        B, N, D = x.shape
        x=x[:, :, :3]
        # print('SampleGroupMoudle x', x.shape)
        volume=calculate_point_cloud_volume(x)
        # print('SampleGroupMoudle volume', volume)
        keypoint_num=int(torch.max(self.density*volume, 0)[0][0])
        print('SampleGroupMoudle keypoint_num', keypoint_num)
        fps_idx = farthest_point_sample(x, keypoint_num)  # [B, keypoint_num, D]

        return fps_idx





class LayeredExtractMoudle_part(nn.Module):
    def __init__(self, mlps,npoint,density_list,sample_num,radiu, small, large,block_num=2,all_extra=False):
        super(LayeredExtractMoudle_part, self).__init__()
        self.all_extra=all_extra
        if self.all_extra:
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = mlps[0]
            for out_channel in mlps[1:]:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            return
        self.fem=FreqExtractMoudle()
        density_total=0
        for density in density_list:
            density_total+=density
        self.sgm_list=[SampleGroupMoudle_part(npoint*density_list[i]/density_total) for i in range(block_num)]
        self.block_num=block_num
        self.density_list=density_list
        self.sample_num=sample_num
        self.radiu=radiu
        self.small=small
        self.large=large
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlps[0]
        for out_channel in mlps[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    # @calculate_time
    def forward(self, x, feature):
        """
        Input:
            x: input pointsXYZ, [B, N1, 3]
            feature: input points feature, [B, N1, C1]
        Return:
            new_xyz: sampled points position data, [B, N2, 3]
            new_feature: sample points feature data, [B, N2, C2]
        """
        B, N, C = x.size()
        if self.all_extra:
            pointfeat = feature.permute(0, 2, 1).unsqueeze(3)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                pointfeat = F.relu(bn(conv(pointfeat)))
            new_feature = torch.max(pointfeat, 2)[0]
            new_feature = new_feature.permute(0, 2, 1)
            new_xyz=torch.mean(x,1).unsqueeze(1)
            return new_xyz, new_feature
        freq_vect=self.fem(x, self.small, self.large)
        sorted_indices = torch.argsort(freq_vect, dim=1)
        block_size = int(N // self.block_num)
        split_indices = torch.split(sorted_indices, block_size, dim=1)
        split_indices=split_indices[:self.block_num]
        points_concat_index=[]
        for i, indices in enumerate(split_indices):
            pointindex=self.sgm_list[i](index_points(x,indices))
            index= index_points(indices.unsqueeze(-1),pointindex).squeeze(-1)
            points_concat_index.append(index)
        keypoint_with_n_index=torch.cat(points_concat_index, dim=1)
        new_xyz = index_points(x, keypoint_with_n_index)
        idx = query_ball_point(self.radiu, self.sample_num, x, new_xyz)
        pointfeat=index_points(feature, idx)# [B, new_xyz_num, sample_num, C]
        pointfeat = pointfeat.permute(0, 3, 2, 1)  # [B, C, nsample,new_xyz_num]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            pointfeat =  F.relu(bn(conv(pointfeat)))
        new_feature = torch.max(pointfeat, 2)[0]
        new_feature = new_feature.permute(0, 2, 1)
        return new_xyz, new_feature



class LayeredExtractMoudle(nn.Module):
    def __init__(self, mlps,density_list,sample_num,radiu, small, large,block_num=4,all_extra=0):
        super(LayeredExtractMoudle, self).__init__()
        self.all_extra=all_extra
        if self.all_extra!=0:
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = mlps[0]
            self.radiu =radiu
            self.sample_num=sample_num
            for out_channel in mlps[1:]:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            return
        self.fem=FreqExtractMoudle()
        self.sgm_list=[SampleGroupMoudle(density_list[i]) for i in range(block_num)]
        self.block_num=block_num
        self.density_list=density_list
        self.sample_num=sample_num
        self.radiu=radiu
        self.small=small
        self.large=large
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlps[0]
        for out_channel in mlps[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    # @calculate_time
    def forward(self, x, feature=None):
        """
        Input:
            x: input pointsXYZ, [B, N1, 3]
            feature: input points feature, [B, N1, C1]
        Return:
            new_xyz: sampled points position data, [B, N2, 3]
            new_feature: sample points feature data, [B, N2, C2]
        """
        # print('LayeredExtractMoudle x', x.shape )
        B, N, C = x.size()
        if self.all_extra==1:
            pointfeat = feature.permute(0, 2, 1).unsqueeze(-1)  # [B,N, C]
            # print("LayeredExtractMoudle pointfeat",pointfeat.shape)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                pointfeat = F.relu(bn(conv(pointfeat)))
            new_feature = torch.max(pointfeat, 2)[0]
            new_feature = new_feature.permute(0, 2, 1)
            # print(f" global_feature:{new_feature.shape} ")
            return None, new_feature
        elif self.all_extra!=0:
            new_xyz=index_points(x,farthest_point_sample(x, self.all_extra))

            idx = query_ball_point(self.radiu, self.sample_num, x, new_xyz)
            pointfeat = index_points(feature, idx)  # [B, new_xyz_num, sample_num, C]
            pointfeat = pointfeat.permute(0, 3, 2, 1)  # [B, C, nsample,new_xyz_num]
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                pointfeat = F.relu(bn(conv(pointfeat)))
            new_feature = torch.max(pointfeat, 2)[0]
            new_feature = new_feature.permute(0, 2, 1)
            return new_xyz, new_feature
        freq_vect=self.fem(x, self.small, self.large)
        # print('LayeredExtractMoudle freq_vect', freq_vect.shape)
        # print(freq_vect[0])
        sorted_indices = torch.argsort(freq_vect, dim=1)
        block_size = int(N // self.block_num)
        # print(block_size)
        # remaining_elements = N % self.block_num
        split_indices = torch.split(sorted_indices, block_size, dim=1)
        # if remaining_elements != 0:
        split_indices=split_indices[:self.block_num]
        # print('LayeredExtractMoudle split_indices',len(split_indices),str([i.shape for i in split_indices]))
        # 如果剩余元素不为零，则将剩余元素重复添加到最后一个块中
        # if remaining_elements != 0:
        #     last_index = block_size * self.block_num
        #     last_indices = sorted_indices[:, last_index:]
        #     repeated_indices = last_indices.repeat(1, block_size + remaining_elements, 1)
        #     split_indices[-1] = torch.cat((split_indices[-1], repeated_indices), dim=1)

        # pcd1 = open3d.geometry.PointCloud()
        # point1 = split_tensors[0].cpu().numpy()[0]
        # pcd1.points = open3d.utility.Vector3dVector(point1)
        #
        # pcd2 = open3d.geometry.PointCloud()
        # point2 = split_tensors[1].cpu().numpy()[0]
        # pcd2.points = open3d.utility.Vector3dVector(point2)
        #
        # pcd3 = open3d.geometry.PointCloud()
        # point3 = split_tensors[2].cpu().numpy()[0]
        # pcd3.points = open3d.utility.Vector3dVector(point3)
        #
        # pcd4 = open3d.geometry.PointCloud()
        # point4 = split_tensors[3].cpu().numpy()[0]
        # pcd4.points = open3d.utility.Vector3dVector(point4)
        #
        # # 创建四个颜色数组
        # color1 = np.zeros((len(point4), 3))
        # color1[:, 0] = 1.0  # 设置为红色
        #
        # color2 = np.zeros((len(point4), 3))
        # color2[:, 1] = 1.0  # 设置为绿色
        #
        # color3 = np.zeros((len(point4), 3))
        # color3[:, 2] = 1.0  # 设置为蓝色
        #
        # color4 = np.zeros((len(point4), 3))
        # color4[:, 0] = 1.0  # 设置为黄色
        # color4[:, 1] = 1.0  # 设置为黄色
        # pcd1.colors = open3d.utility.Vector3dVector(color1)
        # pcd2.colors = open3d.utility.Vector3dVector(color2)
        # pcd3.colors = open3d.utility.Vector3dVector(color3)
        # pcd4.colors = open3d.utility.Vector3dVector(color4)
        # open3d.visualization.draw_geometries([pcd1,pcd2,pcd3,pcd4])
        # 打印块数和每个块的形状
        points_concat_index=[]
        # print(f"Number of Blocks: {len(split_tensors)}")
        for i, indices in enumerate(split_indices):
            block_mean=index_points(freq_vect.unsqueeze(-1),indices).squeeze(-1).float().mean(dim=1)
            # print(f"Block {i + 1} Mean:{block_mean[0]} ")
            pointindex=self.sgm_list[i](index_points(x,indices))
            index= index_points(indices.unsqueeze(-1),pointindex).squeeze(-1)
            # print(index.shape)
            points_concat_index.append(index)
        keypoint_with_n_index=torch.cat(points_concat_index, dim=1)
        writer.add_scalar('choose',keypoint_with_n_index.shape[1] )
        # print('LayeredExtractMoudle keypoint_with_n_index', keypoint_with_n_index.shape)
        new_xyz = index_points(x, keypoint_with_n_index)
        # print('LayeredExtractMoudle new_xyz', new_xyz.shape)
        idx = query_ball_point(self.radiu, self.sample_num, x, new_xyz)
        # new_xyz_ball = index_points(x, idx)  # [B, npoint, nsample, C]
        # print('LayeredExtractMoudle new_xyz_ball', new_xyz_ball.shape)
        if feature !=None:
            pointfeat=index_points(feature, idx)# [B, new_xyz_num, sample_num, C]
        else:
            pointfeat = index_points(x, idx)# [B, new_xyz_num, sample_num, 3]
        # print(f"LayeredExtractMoudle pointfeat shape:{pointfeat.shape} ")
        pointfeat = pointfeat.permute(0, 3, 2, 1)  # [B, C, nsample,new_xyz_num]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            pointfeat =  F.relu(bn(conv(pointfeat)))
        new_feature = torch.max(pointfeat, 2)[0]
        new_feature = new_feature.permute(0, 2, 1)
        # print(f"LayeredExtractMoudle new_xyz {new_xyz.shape} new_feature:{new_feature.shape} ")
        return new_xyz, new_feature

class LayeredExtractMoudle_seg(nn.Module):
    def __init__(self, mlps,npoint,density_list,sample_num,radiu, small, large,block_num=2,all_extra=False):
        super(LayeredExtractMoudle_seg, self).__init__()
        self.all_extra=all_extra
        if self.all_extra:
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            self.sample_num = sample_num
            self.radiu = radiu
            last_channel = mlps[0]
            for out_channel in mlps[1:]:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            return
        self.fem=FreqExtractMoudle()
        density_total=0
        for density in density_list:
            density_total+=density
        self.sgm_list=[SampleGroupMoudle_part(npoint*density_list[i]/density_total) for i in range(block_num)]
        self.block_num=block_num
        self.density_list=density_list
        self.sample_num=sample_num
        self.radiu=radiu
        self.small=small
        self.large=large
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlps[0]
        for out_channel in mlps[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self, x, feature):
        """
        Input:
        x: input pointsXYZ, [B, N1, 3]
        feature: input points feature, [B, N1, C1]
        Return:
        new_xyz: sampled points position data, [B, N2, 3]
        new_feature: sample points feature data, [B, N2, C2]
        """

        B, N, C = x.size()
        if self.all_extra:
            pointfeat = feature.permute(0, 2, 1).unsqueeze(3)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                pointfeat = F.relu(bn(conv(pointfeat)))
            new_feature = torch.max(pointfeat, 2)[0]
            new_feature = new_feature.permute(0, 2, 1)
            new_xyz = torch.mean(x, 1).unsqueeze(1)
            return new_xyz, new_feature
        freq_vect = self.fem(x, self.small, self.large)
        sorted_indices = torch.argsort(freq_vect, dim=1)
        block_size = int(N // self.block_num)
        split_indices = torch.split(sorted_indices, block_size, dim=1)
        split_indices = split_indices[:self.block_num]
        points_concat_index = []
        for i, indices in enumerate(split_indices):
            pointindex = self.sgm_list[i](index_points(x, indices))
            index = index_points(indices.unsqueeze(-1), pointindex).squeeze(-1)
            points_concat_index.append(index)
        keypoint_with_n_index = torch.cat(points_concat_index, dim=1)
        new_xyz = index_points(x, keypoint_with_n_index)
        idx = query_ball_point(self.radiu, self.sample_num, x, new_xyz)
        pointfeat = index_points(feature, idx)  # [B, new_xyz_num, sample_num, C]
        pointfeat = pointfeat.permute(0, 3, 2, 1)  # [B, C, nsample,new_xyz_num]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            pointfeat = F.relu(bn(conv(pointfeat)))
        new_feature = torch.max(pointfeat, 2)[0]
        new_feature = new_feature.permute(0, 2, 1)
        return new_xyz, new_feature

class UNsamplingMoudle(nn.Module):
    def __init__(self,  mlps):
        super(UNsamplingMoudle, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = mlps[0]
        for out_channel in mlps[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, x1,x2 ,feaure1, feature2):
        """
        Input:
        # x1: input points position data, [B, N1,3]
        # x2: sampled input points position data, [B, N2,3]
        points1: input points data, [B, N1, C1]
        points2: input points data, [B, N2, C2]
        Return:
        new_points: upsampled points data, [B, N1,C3]
        """

        B, N, C = feaure1.shape
        _, S, _ = feature2.shape
        if S == 1 :
            interpolated_points = feature2.repeat(1, N, 1)
        else:
            dists = square_distance(x1, x2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3] # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(feature2, idx) * weight.view(B, N, 3, 1), dim=2)
        new_feaure1 = torch.cat([feaure1, interpolated_points], dim=-1)

        new_feaure1 = new_feaure1.permute(0, 2, 1).unsqueeze(-1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feaure1 = F.relu(bn(conv(new_feaure1)))
        new_feaure1 = new_feaure1.squeeze(-1).permute(0, 2, 1)
        # print("new_feaure1", new_feaure1.shape)

        return new_feaure1
