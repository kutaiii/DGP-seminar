import open3d as o3d
import numpy as np

# Harris3D
def harris3d(point_cloud:o3d.geometry.PointCloud, radius=1, max_nn=1000, threshold=0.001):
    """
    Harris3D corner detection algorithm.
    
    Parameters:
        point_cloud (open3d.geometry.PointCloud): Input point cloud.
        radius (float): Radius for neighborhood search.
        threshold (float): Threshold for corner response function.
    
    Returns:
        numpy.ndarray: Indices of detected corners.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    # Compute Harris response
    harris_response = np.zeros(len(point_cloud.points))
    is_corner = np.zeros(len(point_cloud.points), dtype=bool)

    for i in range(len(point_cloud.points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], max_nn)
        selected_points = point_cloud.select_by_index(idx)
        selected_points.points = selected_points.normals
        # Compute covariance matrix
        _, cov_matrix = selected_points.compute_mean_and_covariance()
        harris_response[i] = np.linalg.det(cov_matrix) / (np.trace(cov_matrix) ** 2 + 1e-10)
        print(f"Point {i}: Harris response = {harris_response[i]}")
        if harris_response[i] > threshold:
            is_corner[i] = True

    # Non-maximum suppression
    corners = []
    for i in range(len(point_cloud.points)):
        if is_corner[i]:
            corners.append(i)
            # Suppress neighbors
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point_cloud.points[i], radius)
            for j in idx:
                is_corner[j] = False
    return corners

# Instrinsic Shape Signatures (ISS)
def iss(points, normals, radius=0.1, threshold=0.1):
    """
    Intrinsic Shape Signatures (ISS) corner detection algorithm.
    
    Parameters:
        points (numpy.ndarray): Nx3 array of 3D points.
        normals (numpy.ndarray): Nx3 array of normals corresponding to the points.
        radius (float): Radius for neighborhood search.
        threshold (float): Threshold for corner response function.
    
    Returns:
        numpy.ndarray: Indices of detected corners.
    """
    # Compute the ISS response
    iss_response = np.zeros(len(points))
    for i in range(len(points)):
        iss_response[i] = np.linalg.norm(normals[i])  # Placeholder for actual ISS computation
    
    # Thresholding
    corners = np.where(iss_response > threshold)[0]
    
    return corners.tolist()

# utils
# Non maximum suppression
def non_maximum_suppression(points, scores, radius=0.1):
    """
    Non-maximum suppression for corner detection.
    
    Parameters:
        points (numpy.ndarray): Nx3 array of 3D points.
        scores (numpy.ndarray): Scores for each point.
        radius (float): Radius for suppression.
    
    Returns:
        numpy.ndarray: Indices of suppressed corners.
    """
    suppressed = []
    for i in range(len(points)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < radius and scores[j] > scores[i]:
                suppressed.append(j)
    return np.setdiff1d(np.arange(len(points)), suppressed).tolist()

def main(harris=False, iss=False):
    # Load a point cloud
    pcd = o3d.io.read_point_cloud("./pcd/bun_zipper.ply")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    def draw_corners(corners, color=[1, 0, 0]):
        # Draw spheres at the corners
        spheres = o3d.geometry.TriangleMesh()
        for corner in corners.points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            sphere.translate(corner)
            sphere.paint_uniform_color(color)
            spheres += sphere
        return spheres

    # Harris3D corner detection
    if harris:
        corners_harris = harris3d(pcd, radius=10, max_nn=10, threshold=0.005)
        print(f"Harris3D corners: {len(corners_harris)}")
        corners_harris = pcd.select_by_index(corners_harris)
        harris_corners_mesh = draw_corners(corners_harris, color=[1, 0, 0])
        o3d.visualization.draw_geometries([pcd, harris_corners_mesh], window_name="Harris3D Corners")

    if iss:
        print("Computing ISS corners...")
        keypoint = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius=0.005, non_max_radius=0.005, gamma_21=0.6, gamma_32=0.6)
        keypoint = draw_corners(keypoint, color=[0, 1, 0])
        o3d.visualization.draw_geometries([pcd, keypoint], window_name="ISS Corners")
        o3d.io.write_triangle_mesh("bun_corners.ply", keypoint)





if __name__ == "__main__":
    main(iss=True)