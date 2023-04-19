import json
import pickle
import open3d as o3d
import random
import numpy as np
import numpy.ma as ma
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation


from dataset.bop_utils import *

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

    def __init__(self, args, mode='train'):
        super(LMODataset, self).__init__()
        # arguments
        self.args = args
        # root dir
        self.base_dir = args.data_folder + '/linemod/'
        # whether to do data augmentation
        self.data_augmentation = args.data_augmentation
        # original benchmark or rotated benchmark
        self.rotated = args.rotated
        # factor used to control the maximum rotation during data augmentation
        self.rot_factor = args.rot_factor
        # maximum noise used in data augmentation
        self.augment_noise = args.augment_noise
        # the maximum number allowed in each single frame of point cloud
        self.points_limit = args.points_limit
        # can be in ['train', 'test']
        self.mode = mode
        # view point is the (0,0,0) of cad model
        self.view_point = np.array([0., 0., 0.])  
        # radius used to find the nearest neighbors
        self.corr_radius = 0.001
        self.gt_vis = args.gt_vis
        # loaded data
        self.pickle_file = self.base_dir + 'cache/lm.pkl'
        if args.data_from_pkl:
            with open(self.pickle_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = self.get_dataset_from_path()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


    def get_dataset_from_path(self):

        data = []
        model_root = self.base_dir + 'models'
        frame_root = self.base_dir + self.mode


        model_files = list(Path(model_root).glob('*.ply'))
        obj_num = len(model_files)
        
        for obj_id in tqdm(range(obj_num)):
            

            model_path = str(model_files[obj_id])

            src_pcd_, _ = sample_point_from_mesh(model_path, samples=10000)
            src_pcd = src_pcd_ / 1000

            model_id = str(obj_id + 1).zfill(6)
            frame_path = frame_root + '/' + model_id
            depth_path = frame_path + '/depth'
            mask_path = frame_path + '/mask_visib'

            gt_path = '{0}/scene_gt.json'.format(frame_path)
            cam_path = '{0}/scene_camera.json'.format(frame_path)
            

            xmap = np.array([[j for i in range(640)] for j in range(480)])
            ymap = np.array([[i for i in range(640)] for j in range(480)])


            depth_files = list(Path(depth_path).glob('*.png'))
            mask_files = list(Path(mask_path).glob('*.png'))
            frame_num = len(depth_files)
            
            rand_frame =random.randint(1, frame_num)
            for frame_id in tqdm(range(frame_num)):
                cam_cx, cam_cy, cam_fx, cam_fy = get_camera_info(cam_path, frame_id)
                rot, trans = get_gt(gt_path, frame_id) 

                depth = np.array(Image.open(str(depth_files[frame_id])))
                vis_mask = np.array(Image.open(str(mask_files[frame_id])))
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

                src_pcd = resize_pcd(src_pcd_, self.points_limit)
                tgt_pcd = resize_pcd(tgt_pcd, self.points_limit)

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

                corr, coverage = get_corr(tgt_pcd, src_pcd, rot, trans, self.corr_radius)

                if self.gt_vis and frame_id == rand_frame:
                    gt_visualisation(src_pcd, tgt_pcd, trans, rot, corr)

                frame_data = {
                    'obj_id': obj_id,
                    'frame_id': frame_id,
                    'src_pcd': src_pcd.astype(np.float32),
                    'tgt_pcd': tgt_pcd.astype(np.float32),
                    'rot': rot.astype(np.float32),
                    'trans': trans.astype(np.float32),
                    'corr': corr.astype(np.int32),
                    'coverage': coverage
                }
                data.append(frame_data)
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(data, f)
        return data

def gt_visualisation(src_pcd, tgt_pcd, trans, rot, corr):
    shift = trans + 0.1
    src_t = transformation_pcd(src_pcd, rot, shift)
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(src_t)
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(tgt_pcd)

    points = []
    lines = []
    for i in range(corr.shape[0]):
        src_point = src_t[corr[i, 1]]
        tgt_point = tgt_pcd[corr[i, 0]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(points)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])
    o3d.visualization.draw_geometries([pcd_model, pcd_frame, line])



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