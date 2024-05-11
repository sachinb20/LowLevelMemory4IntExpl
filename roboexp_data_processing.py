

import pickle
import open3d as o3d
import numpy as np


with open('observations_0.pkl', 'rb') as f:
    data = pickle.load(f)

depth_data = data["wrist_2"]["depths"]
c2w = data["wrist_2"]["c2w"]
position = data["wrist_2"]["position"]
mask = np.logical_and(depth_data > 0, depth_data < 1)
mask_broadcasted = np.expand_dims(mask, axis=-1)

# Multiply mask with position data
# position = mask_broadcasted * position
position_data = position.reshape(-1, 3)
pcd_position = o3d.geometry.PointCloud()
pcd_position.points = o3d.utility.Vector3dVector(position_data)

# Visualize both point clouds
o3d.visualization.draw_geometries([pcd_position])

# print(np.max(depth_data))

# with open('observations_1_updated.pkl', 'rb') as f:
#     data = pickle.load(f)

# depth_data = data["wrist_0"]["depths"]
# c2w = data["wrist_0"]["c2w"]
# position = data["wrist_0"]["position"]

# print(np.max(depth_data))