from common import sample_point_from_mesh, get_bbox,mask_to_bbox
import open3d as o3d
import numpy as np
from PIL import Image
from common import normal_redirect
import os.path
import torch
import json
import numpy.ma as ma
import torch.utils.data as data
from scipy.spatial.transform import Rotation
import random
from bop_utils import *

class LMODataset(data.Dataset):
    '''
    Load subsampled coordinates, relative rotation and translation
    Output (torch.Tensor):
    src_pcd: (N, 3) source point cloud
    tgt_pcd: (M, 3) target point cloud
    src_node_xyz: (n, 3) nodes sparsely sampled from source point cloud
    tgt_node_xyz: (m, 3) nodes sparsely sampled from target point cloud
    rot: (3, 3)
    trans: (3, 1)
    correspondences: (?, 3)
    '''

    def __init__(self, args, infos, mode='train'):
        super(LMODataset, self).__init__()
        # arguments
        self.args = args
        # information of data
        self.infos = infos
        # root dir
        self.base_dir = args.data_folder + '/linemod/'
        # whether to do data augmentation
        self.data_augmentation = args.data_augmentation
        # original benchmark or rotated benchmark
        self.rotated = args.rotated
        # factor used to control the maximum rotation during data augmentation
        self.rot_factor = args.rot_factor
        # maximum noise used in data augmentation
        self.augment_noise = args.data_augmentation
        # the maximum number allowed in each single frame of point cloud
        self.points_limit = args.points_limit
        # can be in ['train', 'test']
        self.mode = mode


        if self.mode == 'train':
            train_ls = np.loadtxt(self.infos)
            self.syn_ls = [int(x) for x in train_ls]
        elif self.mode == 'val':
            val_ls = np.loadtxt(self.infos)
            self.syn_ls = [int(x) for x in val_ls]
        else:
            val_ls = np.loadtxt(self.infos)
            self.syn_ls = [int(x) for x in val_ls]

        self.view_point = np.array([0., 0., 0.])  # view point is the (0,0,0) of cad model

        self.data = get_dataset_from_path(self.base_dir, self.mode)


    def __len__(self):
        return len(self.syn_ls)

    def __getitem__(self, index):

        # get gt transformation
        obj_id = 6
        sample_points_from_model =10000
        img_id = self.syn_ls[index]


        depth_path = frame_root+'/000006/depth/' + str(img_id).zfill(6) + '.png'
        mask_path = frame_root+'/000006/mask_visib/' + str(img_id).zfill(6) + '_000000.png'

        gt_file = open('{0}/{1}/scene_gt.json'.format(frame_root, str(obj_id).zfill(6)), 'r', encoding='utf-8')
        cam_file = open('{0}/{1}/scene_camera.json'.format(frame_root, str(obj_id).zfill(6)), 'r', encoding='utf-8')

        gt = json.load(gt_file)[str(img_id)][0]
        cam = json.load(cam_file)[str(img_id)]
        cam_k = np.array(cam['cam_K']).reshape(3, 3)
        cam_cx = cam_k[0, 2]
        cam_cy = cam_k[1, 2]
        cam_fx = cam_k[0, 0]
        cam_fy = cam_k[1, 1]

        xmap = np.array([[j for i in range(640)] for j in range(480)])
        ymap = np.array([[i for i in range(640)] for j in range(480)])

        depth = np.array(Image.open(depth_path))
        vis_mask = np.array(Image.open(mask_path))
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(vis_mask, np.array(255)))
        mask = mask_label * mask_depth

        rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask))
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        tgt_pcd = cloud / 1000.0

        src_pcd_, src_normals = sample_point_from_mesh(model_root, obj_id,sample_points_from_model)
        src_pcd = src_pcd_ / 1000

        rot = np.array(gt['cam_R_m2c']).reshape(3, 3)
        trans = np.array(gt['cam_t_m2c']) / 1000
        target = np.dot(src_pcd, rot.T)
        target = np.add(target, trans)

        ##################################################################################################
        # if we get too many points, we do random down-sampling
        if src_pcd.shape[0] > self.points_limit:
            idx = np.random.permutation(src_pcd.shape[0])[:self.points_lim]
            src_pcd = src_pcd[idx]

        if tgt_pcd.shape[0] > self.points_limit:
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.points_lim]
            tgt_pcd = tgt_pcd[idx]

        ##################################################################################################
        # whether to augment data for training / to rotate data for testing
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
            # add noise
            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
        # wheter test on rotated benchmark
        elif self.rotated:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T

                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T

                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)
        else:
            pass

        if (trans.ndim == 1):
            trans = trans[:, None]

        ##################################################################################################

        return src_pcd.astype(np.float32), tgt_pcd.astype(np.float32), \
               rot.astype(np.float32), trans.astype(np.float32),\
               src_pcd.astype(np.float32), None




def debug_data():
    obj_id = 6
    img_id = 0
    sample_points_from_model=10000

    model_root = '/home/hjw/projects/RIGAv2_non_rigid/data/lmo/lmo_models'
    frame_root = '/home/hjw/projects/RIGAv2_non_rigid/data/lmo/lmo_train/train'

    depth_path = '/home/hjw/bop-dataset/lmo/lmo_train/train/000006/depth/' + str(img_id).zfill(6) + '.png'
    mask_path = '/home/hjw/bop-dataset/lmo/lmo_train/train/000006/mask_visib/' + str(img_id).zfill(6) + '_000000.png'

    info_file = open('{0}/{1}/scene_gt_info.json'.format(frame_root, str(obj_id).zfill(6)), 'r', encoding='utf-8')
    gt_file = open('{0}/{1}/scene_gt.json'.format(frame_root, str(obj_id).zfill(6)), 'r', encoding='utf-8')
    cam_file = open('{0}/{1}/scene_camera.json'.format(frame_root, str(obj_id).zfill(6)), 'r', encoding='utf-8')

    info = json.load(info_file)[str(img_id)]
    gt = json.load(gt_file)[str(img_id)][0]
    cam = json.load(cam_file)[str(img_id)]
    cam_k = np.array(cam['cam_K']).reshape(3, 3)
    cam_cx = cam_k[0, 2]
    cam_cy = cam_k[1, 2]
    cam_fx = cam_k[0, 0]
    cam_fy = cam_k[1, 1]

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    depth = np.array(Image.open(depth_path))
    vis_mask = np.array(Image.open(mask_path))
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(vis_mask, np.array(255)))
    mask = mask_label * mask_depth

    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask))
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    cam_scale = 1.0
    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)
    cloud = cloud / 1000.0

    model_points = sample_point_from_mesh(model_root, obj_id,sample_points_from_model) / 1000
    target_r = np.array(gt['cam_R_m2c']).reshape(3, 3)
    target_t = np.array(gt['cam_t_m2c']) / 1000
    target = np.dot(model_points, target_r.T)
    target = np.add(target, target_t)

    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(target)
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd_frame, pcd_model])
    print()

if __name__ == '__main__':
    debug_data()