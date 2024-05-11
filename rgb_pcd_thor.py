

import math
import numpy as np
import open3d as o3d
import pickle 

def to_rad(th):
    return th*math.pi / 180

def example():
    width, height = 640, 480
    fov = 90
    # controller = Controller(scene="FloorPlan1",
    #                         width=width,
    #                         height=height,
    #                         fieldOfView=fov,
    #                         renderDepthImage=True)
    # controller.step(action="RotateLeft", degrees=45)
    # event = controller.step(action="Pass")

    # Convert fov to focal length
    focal_length = 0.5 * width * math.tan(to_rad(fov/2))

    # camera intrinsics
    fx, fy, cx, cy = (focal_length, focal_length, width/2, height/2)
    with open('observations_3_updated.pkl', 'rb') as f:
        data = pickle.load(f)

    # Obtain point cloud
    color = o3d.geometry.Image((data["wrist_0"]["rgb"]*255).astype(np.uint8))
    d_float = data["wrist_0"]["depths"]
    # d_float = d.astype(np.float32)
    d_float /= np.max(d_float)
    depth = o3d.geometry.Image(d_float)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                              depth_scale=1.0,
                                                              depth_trunc=0.7,
                                                              convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    example()