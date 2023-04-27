import json
import cv2
import torch
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from focal_loss.focal_loss import FocalLoss



def sample_point_from_mesh(model_root,samples):
    r"""Sample given number of points from a mesh readed from path.
    """
    mesh = o3d.io.read_triangle_mesh(model_root)
    pcd = mesh.sample_points_uniformly(number_of_points=samples)
    scale_factor = 0.001
    pcd.scale(scale_factor,(0, 0, 0))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals

def get_bbox(bbox):
    r"""Get bounding box from a mask.
    Return coordinates of the bounding box [x_min, y_min, x_max, y_max]
    """
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

    rmin, rmax, cmin, cmax = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
    rmin = max(rmin, 0)
    rmax = min(rmax, 479)
    cmin = max(cmin, 0)
    cmax = min(cmax, 639)
    r_b = rmax - rmin
    c_b = cmax - cmin

    for i in range(len(border_list) - 1):
        if r_b > border_list[i] and r_b < border_list[i + 1]:
            r_b = border_list[i + 1]
            break
    for i in range(len(border_list) - 1):
        if c_b > border_list[i] and c_b < border_list[i + 1]:
            c_b = border_list[i + 1]
            break

    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, 480)
    cmax = min(cmax, 640)

    return rmin, rmax, cmin, cmax


def mask_to_bbox(mask):
    r"""Get bounding box from a mask.
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return list(bbox)

def get_gt(gt_file, frame_id):
    r"""Get ground truth pose from a ground truth file.
    Return rotation matrix and translation vector
    """
    with open(gt_file, 'r') as file:
        gt = json.load(file)[str(frame_id)][0]
    rot = np.array(gt['cam_R_m2c']).reshape(3, 3)
    trans = np.array(gt['cam_t_m2c']) / 1000
    return rot, trans

def get_camera_info(cam_file, frame_id):
    r"""Get camera intrinsics from a camera file.
    Return camera center, focal length
    """
    with open(cam_file, 'r') as file:
        cam = json.load(file)[str(frame_id)]
    cam_k = np.array(cam['cam_K']).reshape(3, 3)
    cam_cx = cam_k[0, 2]
    cam_cy = cam_k[1, 2]
    cam_fx = cam_k[0, 0]
    cam_fy = cam_k[1, 1]
    return cam_cx, cam_cy, cam_fx, cam_fy

def resize_pcd(pcd, points_limit):
    r"""Resize a point cloud to a given number of points.
    """
    if pcd.shape[0] > points_limit:
        idx = np.random.permutation(pcd.shape[0])[:points_limit]
        pcd = pcd[idx]
    return pcd

def transformation_pcd(pcd, rot, trans):
    r"""Transform a point cloud with a rotation matrix and a translation vector.
    """
    pcd_t = np.dot(pcd, rot.T)
    pcd_t = np.add(pcd_t, trans.T)
    return pcd_t

def get_corr(tgt_pcd, src_pcd, rot, trans, radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_t = transformation_pcd(src_pcd, rot, trans)
    src_tree = cKDTree(src_t)
    indices_list = src_tree.query_ball_point(tgt_pcd, radius)
    corr = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.int32,
    )
    coverage = corr.shape[0] / tgt_pcd.shape[0]
    return corr, coverage

def get_corr_matrix(corr, tgt_len, src_len):
    r"""Get a correspondence matrix from a correspondence array.
    Return correspondence matrix [tgt_len, src_len]
    """
    corr_matrix = np.full((tgt_len, src_len), -1.0, dtype=np.float32)
    corr_matrix[corr[:, 0], corr[:, 1]] = 1.0
    return corr_matrix

def get_corr_src_pcd(corr, src_pcd):
    r"""Get the source point cloud of the correspondences.
    Return source point cloud of the correspondences
    """
    return src_pcd[corr[:, 1]]

def get_corr_similarity(corr_matrix, gt_corr_matrix):
    r"""Get the cosine distance similarity between the correspondence matrix 
    and the ground truth correspondence matrix.
    Return cosine distance similarity
    """
    corr_matrix = corr_matrix.astype(np.float32)
    gt_corr_matrix = gt_corr_matrix.astype(np.float32)
    num = np.dot(corr_matrix, gt_corr_matrix.T)  
    denom = np.linalg.norm(corr_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(gt_corr_matrix, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def focal_loss(y_true, y_pred, gamma=2):
    """
    Focal loss function for binary classification
    """
    criterion = FocalLoss(gamma)
    y_pred = torch.from_numpy(y_pred.flatten())
    y_pred = torch.nn.Sigmoid()(y_pred)
    y_true = (y_true + 1) / 2
    y_true = torch.from_numpy(y_true.flatten().astype(np.int64))
    focal_loss = criterion(y_pred, y_true)
    return focal_loss.numpy()

def get_corr_from_matrix_topk(corr_matrix, k):
    r"""Get the top k correspondences from a correspondence matrix.
    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    corr_indices = corr_matrix.view(-1).topk(k=k, largest=True)[1]
    ref_corr_indices = corr_indices // corr_matrix.shape[1]
    src_corr_indices = corr_indices % corr_matrix.shape[1]
    corr = np.array(
        [(i, j) for i, j in zip(ref_corr_indices, src_corr_indices)],
        dtype=np.int32,
    )
    return corr


def gt_visualisation(src_pcd, tgt_pcd, trans, rot, corr):
    r"""Visualise the ground truth correspondences between two point clouds.
    shift the transformed source point cloud to avoid overlapping
    """
    shift_t = trans + 0.1
    src_t = transformation_pcd(src_pcd, rot, shift_t)
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(src_pcd)
    pcd_model_t = o3d.geometry.PointCloud()
    pcd_model_t.points = o3d.utility.Vector3dVector(src_t)
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(tgt_pcd)

    

    # draw the correspondences between the transformed source and target
    points = []
    lines = []
    for i in range(corr.shape[0]):
        src_t_point = src_t[corr[i, 1]]
        tgt_point = tgt_pcd[corr[i, 0]]
        points.append(src_t_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])

    line_gt = o3d.geometry.LineSet()
    line_gt.points = o3d.utility.Vector3dVector(points)
    line_gt.lines = o3d.utility.Vector2iVector(lines)
    line_gt.colors = o3d.utility.Vector3dVector([[0, 1, 0] for i in range(len(lines))])

    # draw the correspondences between the original source and target
    points = []
    lines = []
    for i in range(corr.shape[0]):
        src_point = src_pcd[corr[i, 1]]
        tgt_point = tgt_pcd[corr[i, 0]]
        points.append(src_point)
        points.append(tgt_point)
        lines.append([i * 2, i * 2 + 1])

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(points)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector([[0, 0.8, 0.2] for i in range(len(lines))])
    o3d.visualization.draw_geometries([pcd_model, pcd_model_t, pcd_frame, line, line_gt])