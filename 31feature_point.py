import open3d as o3d
import numpy as np

# Harris3D
def harris3d(points, normals, window_size=5, threshold=0.1):
    """
    Harris 3D corner detection algorithm.
    
    Parameters:
        points (numpy.ndarray): Nx3 array of 3D points.
        normals (numpy.ndarray): Nx3 array of normals corresponding to the points.
        window_size (int): Size of the window for computing the covariance matrix.
        threshold (float): Threshold for corner response function.
    
    Returns:
        numpy.ndarray: Indices of detected corners.
    """
    # Compute the covariance matrix
    cov_matrix = np.zeros((3, 3))
    for i in range(len(points)):
        cov_matrix += np.outer(normals[i], normals[i])
    
    # Compute the Harris response
    harris_response = np.linalg.det(cov_matrix) - 0.04 * (np.trace(cov_matrix) ** 2)
    
    # Thresholding
    corners = np.where(harris_response > threshold)[0]
    
    return corners.tolist()

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

def main():
    # Load a point cloud
    pcd = o3d.io.read_point_cloud("path_to_your_point_cloud.ply")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Detect corners using Harris3D
    harris_corners = harris3d(points, normals)
    print("Harris3D corners:", harris_corners)

    # Detect corners using ISS
    iss_corners = iss(points, normals)
    print("ISS corners:", iss_corners)