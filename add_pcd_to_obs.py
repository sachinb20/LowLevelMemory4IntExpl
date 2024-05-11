# import imageio.v3 as iio
# import numpy as np
# import matplotlib.pyplot as plt
# import open3d as o3d

# depth_image = iio.imread('depth000114.png')

# # print properties:
# print(f"Image resolution: {depth_image.shape}")
# print(f"Data type: {depth_image.dtype}")
# print(f"Min value: {np.min(depth_image)}")
# print(f"Max value: {np.max(depth_image)}")

# FX_DEPTH = 261.9551696777344	
# FY_DEPTH = 261.9551696777344	
# CX_DEPTH = 311.6794128417969	
# CY_DEPTH = 185.56922912597656

# pcd = []
# height, width = depth_image.shape
# for i in range(height):
#    for j in range(width):
#        z = depth_image[i][j]
#        x = (j - CX_DEPTH) * z / FX_DEPTH
#        y = (i - CY_DEPTH) * z / FY_DEPTH
#        pcd.append([x, y, z])
       
# pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
# pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
# # Visualize:
# o3d.visualization.draw_geometries([pcd_o3d])

# import pickle
# import open3d as o3d
# import numpy as np

# # Load the pickle file
# with open('observations_0.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Extract point cloud data
# point_cloud = data["wrist_0"]["position"]

# # Reshape the point cloud array
# point_cloud_reshaped = point_cloud.reshape(-1, 3)

# # Create Open3D point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud_reshaped)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])


# import pickle
# import open3d as o3d
# import numpy as np

# # Load the pickle file
# with open('observations_1.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Extract depth data and intrinsic camera parameters
# depth_data = data["wrist_0"]["depths"]
# intrinsic_matrix = data["wrist_0"]["intrinsic"]
# position_data = data["wrist_0"]["position"]

# position_data = position_data.reshape(-1, 3)

# # Generate 3D points from depth map using intrinsic parameters
# rows, cols = depth_data.shape
# v, u = np.meshgrid(range(rows), range(cols), indexing='ij')

# # Convert depth from millimeters to meters
# depth_in_meters = depth_data 

# # Stack u, v, 1 to form homogeneous coordinates
# uv1 = np.stack((u, v, np.ones_like(u)), axis=-1)

# # Apply inverse intrinsic matrix to get normalized coordinates
# normalized_coords = np.matmul(np.linalg.inv(intrinsic_matrix), uv1.reshape(-1, 3, 1)).reshape(-1, 3)

# # Scale by depth to get 3D points
# point_cloud_depth = normalized_coords * depth_in_meters.reshape(-1, 1)
# data["wrist_0"]["position"] = point_cloud_depth.reshape((480, 640, 3))

# Create Open3D point cloud object from depth information
# pcd_depth = o3d.geometry.PointCloud()
# pcd_depth.points = o3d.utility.Vector3dVector(point_cloud_depth)
# print(pcd_depth.shape)
# # Create Open3D point cloud object from position information
# pcd_position = o3d.geometry.PointCloud()
# pcd_position.points = o3d.utility.Vector3dVector(position_data)

# # Visualize both point clouds
# o3d.visualization.draw_geometries([pcd_depth])

# import pickle
# import open3d as o3d
# import numpy as np

# # Loop through each observation file
# for i in range(1, 5):
#     filename = f'observations_{i}.pkl'
    
#     # Load the pickle file
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)

#     # Loop through each wrist position in the loaded data
#     for wrist_id, wrist_data in data.items():
#         depth_data = wrist_data["depths"]
#         intrinsic_matrix = wrist_data["intrinsic"]

#         # Generate 3D points from depth map using intrinsic parameters
#         rows, cols = depth_data.shape
#         v, u = np.meshgrid(range(rows), range(cols), indexing='ij')

#         # Convert depth from millimeters to meters
#         depth_in_meters = depth_data  # Convert millimeters to meters

#         # Stack u, v, 1 to form homogeneous coordinates
#         uv1 = np.stack((u, v, np.ones_like(u)), axis=-1)

#         # Apply inverse intrinsic matrix to get normalized coordinates
#         normalized_coords = np.matmul(np.linalg.inv(intrinsic_matrix), uv1.reshape(-1, 3, 1)).reshape(-1, 3)

#         # Scale by depth to get 3D points
#         point_cloud_depth = normalized_coords * depth_in_meters.reshape(-1, 1)

#         # Reshape point cloud to original image dimensions
#         point_cloud_depth = point_cloud_depth.reshape((rows, cols, 3))

#         # Store the computed point cloud in data
#         wrist_data["position"] = point_cloud_depth

#     # Save the updated data back to the pickle file
#     with open(f'observations_{i}_updated.pkl', 'wb') as f:
#         pickle.dump(data, f)


import pickle
import open3d as o3d
import numpy as np

# Loop through each observation file
for i in range(1, 5):
    filename = f'observations_{i}.pkl'
    
    # Load the pickle file
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Loop through each wrist position in the loaded data
    for wrist_id, wrist_data in data.items():
        wrist_data["c2w"] = np.array(wrist_data["c2w"])
        wrist_data["intrinsic"] = np.array(wrist_data["intrinsic"])
        intrinsic_matrix = wrist_data["intrinsic"]
        depth_data = wrist_data["depths"]

        # Generate 3D points from depth map using intrinsic parameters
        rows, cols = depth_data.shape
        v, u = np.meshgrid(range(rows), range(cols), indexing='ij')

        # Convert depth from millimeters to meters
        depth_in_meters = depth_data  # Convert millimeters to meters

        # Stack u, v, 1 to form homogeneous coordinates
        uv1 = np.stack((u, v, np.ones_like(u)), axis=-1)

        # Apply inverse intrinsic matrix to get normalized coordinates
        normalized_coords = np.matmul(np.linalg.inv(intrinsic_matrix), uv1.reshape(-1, 3, 1)).reshape(-1, 3)

        # Scale by depth to get 3D points
        point_cloud_depth = normalized_coords * depth_in_meters.reshape(-1, 1)

        # Reshape point cloud to original image dimensions
        point_cloud_depth = point_cloud_depth.reshape((rows, cols, 3))

        # Store the computed point cloud in data
        wrist_data["position"] = point_cloud_depth

    # Save the updated data back to the pickle file
    with open(f'observations_{i}_updated.pkl', 'wb') as f:
        pickle.dump(data, f)


# with open('observations_0.pkl', 'rb') as f:
#     data = pickle.load(f)

# depth_data = data["wrist_0"]["depths"]
# c2w = data["wrist_0"]["c2w"]
# position = data["wrist_0"]["position"]
# print(type(position))
# # print(c2w[:3, :3])
# # print(position @ c2w[:3, :3].T + c2w[:3, 3])


with open('observations_1_updated.pkl', 'rb') as f:
    data = pickle.load(f)

depth_data = data["wrist_0"]["depths"]
c2w = data["wrist_0"]["c2w"]
position = data["wrist_0"]["position"]
# print(type(c2w))
# print(c2w[:3, :3])
print(data)