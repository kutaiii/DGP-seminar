import open3d as o3d
import numpy as np

# 立方体の点群 100x100x100
cube_points = np.array([[x, y, z] for x in range(100) for y in range(100) for z in range(100)])

mask_inside = np.where=(cube_points[:, 0] >= 1) & (cube_points[:, 0] < 99) & \
    (cube_points[:, 1] >= 1) & (cube_points[:, 1] < 99) & \
    (cube_points[:, 2] >= 1) & (cube_points[:, 2] < 99)

#反転
mask_inside = np.logical_not(mask_inside)
cube_points = cube_points[mask_inside]
cube_colors = np.array([[0.5, 0.5, 0.5] for _ in range(len(cube_points))])  # Red color

cube_pcd = o3d.geometry.PointCloud()
cube_pcd.points = o3d.utility.Vector3dVector(cube_points)
cube_pcd.colors = o3d.utility.Vector3dVector(cube_colors)

o3d.visualization.draw_geometries([cube_pcd])
o3d.io.write_point_cloud("cube.ply", cube_pcd)