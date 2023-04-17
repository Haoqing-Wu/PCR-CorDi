import json
import os
import open3d as o3d
import numpy as np
from pathlib import Path

from collections import defaultdict

def get_dataset_from_path(base_dir, mode):

    model_root = base_dir + 'models'
    frame_root = base_dir + mode

    model_files = list(Path(model_root).glob('*.ply'))
    obj_num = len(model_files)

    # iterate the models and frame of each object
    data = defaultdict(list)
    for id in range(obj_num):
        model_path = str(model_files[id])
        model_pcd, _ = sample_point_from_mesh(model_path, samples=10000)

        model_id = str(id + 1).zfill(6)
        frame_path = frame_root + '/' + model_id
        gt_file = open('{0}/scene_gt.json'.format(frame_path, 'r', encoding='utf-8'))
        cam_file = open('{0}/scene_camera.json'.format(frame_path, 'r', encoding='utf-8'))

        gt = json.load(gt_file)[str(id)][0]
        cam = json.load(cam_file)[str(id)]
        cam_k = np.array(cam['cam_K']).reshape(3, 3)
        cam_cx = cam_k[0, 2]
        cam_cy = cam_k[1, 2]
        cam_fx = cam_k[0, 0]
        cam_fy = cam_k[1, 1]




        depth_path = frame_path + '/depth'
        mask_path = frame_path + '/mask_visib'

        frame_num = len(frame_files)
        for j in range(frame_num):
            frame_file = frame_files[j]
            frame_name = frame_file.stem
            frame_id = int(frame_name.split('_')[0])
            data[model_id].append(frame_id)


def sample_point_from_mesh(model_root,samples):
    mesh = o3d.io.read_triangle_mesh(model_root)
    pcd = mesh.sample_points_uniformly(number_of_points=samples)
    scale_factor = 0.001
    pcd.scale(scale_factor,(0, 0, 0))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    return points, normals

# test of method get_dataset_from_path
if __name__ == "__main__":
    base_dir = './data/linemod/'
    mode = 'train'
    data = get_dataset_from_path(base_dir, mode)
    print(data)