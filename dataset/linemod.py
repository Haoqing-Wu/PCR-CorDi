import os
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
        self.base_dir = args.data_folder + 'linemod/'
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
        # self.view_point = np.array([0., 0., 0.])  
        # radius used to find the nearest neighbors
        self.corr_radius = 0.01
        # overfitting on a single object
        self.overfit = args.overfit
        # loaded data
        self.pickle_file = self.base_dir + 'cache/lm_{0}_{1}.pkl'.format(self.mode, self.points_limit)
        if os.path.exists(self.pickle_file) and not self.args.reload_data:
            with open(self.pickle_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = self.get_dataset_from_path()
        print('Loaded {0} {1} samples'.format(len(self.data), self.mode))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = self.data[index]
        obj_id = data_dict['obj_id']
        frame_id = data_dict['frame_id']
        src_pcd = data_dict['src_pcd']
        tgt_pcd = data_dict['tgt_pcd']
        rot = data_dict['rot']
        trans = data_dict['trans']

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
            # init features
            src_feats = np.ones_like(src_pcd[:, :1])
            tgt_feats = np.ones_like(tgt_pcd[:, :1])

        # downsample the point cloud and find the neighbors

        # generate the correspondences gt
        #corr, coverage = get_corr(tgt_pcd, src_pcd, rot, trans, self.corr_radius)
        #corr_matrix = get_corr_matrix(corr, tgt_pcd.shape[0], src_pcd.shape[0])


        
        return {
            'obj_id': int(obj_id),
            'frame_id': int(frame_id),
            'src_pcd': src_pcd.astype(np.float32),
            'tgt_pcd': tgt_pcd.astype(np.float32),
            'rot': rot.astype(np.float32),
            'trans': trans.astype(np.float32),
            'src_feats': src_feats.astype(np.float32),
            'tgt_feats': tgt_feats.astype(np.float32)    
        }


    def get_dataset_from_path(self):

        data = []
        model_root = self.base_dir + 'models'
        frame_root = self.base_dir + self.mode


        model_files = list(Path(model_root).glob('*.ply'))
        if self.overfit is not None:
            obj_num = self.overfit
        else:
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


            depth_files = {os.path.splitext(os.path.basename(file))[0]:\
                            str(file) for file in Path(depth_path).glob('*.png')}
            mask_files = {os.path.splitext(os.path.basename(file))[0]:\
                            str(file) for file in Path(mask_path).glob('*.png')}
            frames = list(depth_files.keys())
            
            for frame_id in frames:
                cam_cx, cam_cy, cam_fx, cam_fy = get_camera_info(cam_path, int(frame_id))
                rot, trans = get_gt(gt_path, int(frame_id)) 

                depth = np.array(Image.open(str(depth_files[frame_id])))
                vis_mask = np.array(Image.open(str(mask_files[frame_id + '_000000'])))
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
                tgt_pcd = cloud / 1000.0  # scale to meters

                src_pcd = resize_pcd(src_pcd_, self.points_limit)
                tgt_pcd = resize_pcd(tgt_pcd, self.points_limit)

                if (trans.ndim == 1):
                    trans = trans[:, None]
                src_pcd = sort_pcd_from_center(src_pcd)
                tgt_pcd = sort_pcd_from_center(tgt_pcd)

                frame_data = {
                    'obj_id': int(obj_id),
                    'frame_id': int(frame_id),
                    'src_pcd': src_pcd.astype(np.float32),
                    'tgt_pcd': tgt_pcd.astype(np.float32),
                    'rot': rot.astype(np.float32),
                    'trans': trans.astype(np.float32)
                }
                #for i in range (1000):
                data.append(frame_data)
                #break
            
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(data, f)
        return data

