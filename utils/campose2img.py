import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pytorch3d.io import load_objs_as_meshes
from typing import NamedTuple

from gaussian_renderer import render
import json
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos
import cv2
import time
from tqdm import tqdm
from scene.cameras import Camera
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation as Rota
from gaussian_renderer import GaussianModel


def render_set(views, gaussians, pipeline, background):

    res = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        print(rendering.shape)
        res.append(rendering.permute(1, 2, 0).nan_to_num().clamp(min=0, max=1))
    return res

class SimplePipeline:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def wait_for_pose(control_data: np.ndarray):
    while True:
        if control_data[0] >= 0.5:
            cam_trans = control_data[1:4]
            cam_rot = control_data[4:8]
            fx = control_data[8]
            fy = control_data[9]
            width = control_data[10]
            height = control_data[11]
            return cam_trans, cam_rot, fx, fy, width, height
        else:
            pass

def main():
    tmp_data = np.ones((720, 1280, 3, 10), dtype=np.uint8)
    device = "cuda"
    tmp = np.array([[ 0, -1, 0, 0],
                     [ 0, 0,  -1, 0],
                     [ 1, 0,  0, 0],
                     [ 0, 0,  0, 1]])
    tmp = np.linalg.inv(tmp)
    gaussians = GaussianModel(3)
    gaussians.load_ply("/ssd/hanxiaoshen/main/gaussian-splatting/data/new/mix2/gs-output/1/point_cloud/iteration_30000/point_cloud.ply")
    pipeline = SimplePipeline()
    contorl_tmp_data = np.ones((12,), dtype=np.float64)
    data_shm = shared_memory.SharedMemory(create=True, size=tmp_data.nbytes, name="data_psm_08d5dd701")
    control_shm = shared_memory.SharedMemory(create=True, size=contorl_tmp_data.nbytes, name="control_psm_08d5dd701")
    control_data = np.ndarray((12,), dtype=np.float64, buffer=control_shm.buf)
    control_data[0] = 0.0
    print("Data shm name:", data_shm.name)
    print("Control shm name:", control_shm.name)
    try:
        while True:
            camera_translation, camera_orientation, fx, fy, width, height= wait_for_pose(control_data)
            w, x, y, z = camera_orientation
            camera_orientation = np.array([x, y, z, w])
            camera_orientation = Rota.from_quat(camera_orientation).as_matrix()
            W2C = np.eye(4)
            W2C[:3, :3] = camera_orientation
            W2C[:3, 3] = camera_translation
            W2C = W2C @ tmp
            Rt = np.linalg.inv(W2C)
            R = Rt[:3, :3].transpose()
            T = Rt[:3, 3]
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            img_name = "000"
            print(height, width)
            height = int(height)
            width = int(width)
            sample_image = torch.zeros((3, height, width), dtype=torch.float32, device=device)
            cam = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, image=sample_image, gt_alpha_mask=None, image_name=img_name, uid=0, data_device=device)
            cams = [cam]
            with torch.no_grad():
                res = render_set(cams, gaussians, pipeline, torch.tensor([0, 0, 0], dtype=torch.float32, device=device))
            image = res[0].detach().cpu().numpy()
            image = (image*255).astype(np.uint8)
            buffer = np.ndarray(image.shape, dtype=image.dtype, buffer=data_shm.buf)
            buffer[:] = image[:]
            control_data[0] = 0.0
    finally:
        control_shm.close()
        control_shm.unlink()
        data_shm.close()
        data_shm.unlink()
        
        
if __name__ == "__main__":
    main()
    # json_path = "/home/pjlab/main/real2sim/assets/data/mix5/gs-output/1/cameras.json"
    # device = 'cuda'
    # with open(json_path, 'r') as file:
    #     json_data = json.load(file)
    # json_cam_position = np.array(json_data[0]["position"])
    # json_cam_rotation = np.array(json_data[0]["rotation"])
    # uid = json_data[0]["id"]
    # width = json_data[0]["width"]
    # height = json_data[0]["height"]
    # fx = json_data[0]["fx"]
    # fy = json_data[0]["fy"]
    
    # W2C = np.eye(4)
    # W2C[:3, :3] = json_cam_rotation
    # W2C[:3, 3] = json_cam_position
    # Rt = np.linalg.inv(W2C)
    # R = Rt[:3, :3].transpose()
    # T = Rt[:3, 3]
    # FovX = focal2fov(fx, width)
    # FovY = focal2fov(fy, height)
    # img_name = json_data[0]["img_name"]
    # sample_image = torch.zeros((3, height, width), dtype=torch.float32, device=device)
    # cam = Camera(colmap_id=uid, R=R, T=T, FoVx=FovX, FoVy=FovY, image=sample_image, gt_alpha_mask=None, image_name=img_name, uid=uid, data_device=device)
    # cams = [cam]
    
    # gaussians = GaussianModel(3)
    # gaussians.load_ply("/home/pjlab/main/real2sim/assets/data/mix5/gs-output/1/point_cloud/iteration_30000/point_cloud.ply")
    # pipeline = SimplePipeline()
    # res = render_set(cams, gaussians, pipeline, torch.tensor([0, 0, 0], dtype=torch.float32, device=device))
    # img = res[0].detach().cpu().numpy()
    # img = (img*255).astype(np.uint8)
    # # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("img.jpg", img)
