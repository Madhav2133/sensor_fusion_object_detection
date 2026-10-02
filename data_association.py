#!/usr/bin/env python3

# --- std ---
import argparse
from typing import List, Tuple, Dict, Union

# --- installed ---
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from dataloader import load_data
from sensor_fusion import project_lidar_to_image

BOX_USE_MINIMAL = True # Use minimal points to construct boubding box. Gives better results for OBB if set to true

STD_RATIO = 2.0 # Standard deviation ratio for outlier thresholding.
NB_NEIGHBOUR = 20 # Number of neighbors to analyze for each point.
Z_THRESH = 2 # Points within z_median + z_thresh and z_median - z_thresh are considered.
GROUND_THRESH = -1.6 # Points greater than this z value are considered ground points.


def draw_boxes(
    image: np.ndarray, boxes: List, box_colors: Tuple, thickness: int = 2
) -> None:
    """
    Draws bounding boxes on the image.
    """

    image_with_boxes = image.copy()

    # Draw each bounding box with its corresponding color
    for box, color in zip(boxes, box_colors):
        x1, y1, x2, y2 = map(int, box)
        color = tuple((np.array(color) * 255))  # Convert color from [0,1] to [0,255]
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

    # Display image with boxes
    plt.figure(figsize=(15, 8))
    plt.imshow(image_with_boxes)
    plt.axis("off")
    plt.show()


def frustum_culling(
    lidar_points: np.ndarray,
    boxes: List,
    calib: Dict,
    image: np.ndarray,
    filter_points: bool = True,
) -> List:
    """
    Filters LiDAR points that fall within the 2D image bounding boxes (frustum culling).
    """

    filtered_points_list = list()

    # Project LiDAR points into the image plane
    pixel_coords, depths, mask = project_lidar_to_image(
        lidar_points, calib, image, filter_points=filter_points, visualize=True
    )

    # Depth and ground height thresholds for filtering
    for box in boxes:
        x1, y1, x2, y2 = box

        # Create mask for points that project inside the bounding box and are valid
        in_box_mask = (
            (pixel_coords[:, 0] >= x1)
            & (pixel_coords[:, 0] <= x2)
            & (pixel_coords[:, 1] >= y1)
            & (pixel_coords[:, 1] <= y2)
            & mask
        )

        # Remove ground points based on LiDAR z-values
        z_mask = lidar_points[:, 2] > GROUND_THRESH  # Filter out ground points
        combined_mask = in_box_mask & z_mask

        # Filter LiDAR points inside bounding box
        filtered_points = lidar_points[combined_mask]

        # Further refine points using a depth consistency filter
        median_depth = np.median(depths[combined_mask])
        median_mask = (depths[combined_mask] < (median_depth + Z_THRESH)) & (
            depths[combined_mask] > (median_depth - Z_THRESH)
        )

        filtered_points = filtered_points[median_mask]
        filtered_points_list.append(filtered_points)

    return filtered_points_list


def remove_outliers_from_clusters(
    point_clusters, nb_neighbors=NB_NEIGHBOUR, std_ratio=STD_RATIO
):
    """
    Removes outlier points from each LiDAR cluster using statistical outlier removal.
    """
    cleaned_clusters = list()
    for cluster in point_clusters:
        if cluster.shape[0] < 10:
            cleaned_clusters.append(cluster)
            continue

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster[:, :3])

        # Perform statistical outlier removal
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        # Store filtered points (Nx3)
        filtered_points = np.asarray(pcd_clean.points)
        # add reflectance (if available) to keep data consistency (Nx4)
        if cluster.shape[1] == 4:
            filtered_points = np.hstack(
                [
                    filtered_points,
                    np.ones((filtered_points.shape[0], 1)) * np.mean(cluster[:, 3]),
                ]
            )
        cleaned_clusters.append(filtered_points)
    return cleaned_clusters


def perform_data_association(
    image: np.ndarray,
    calib: Dict,
    lidar: np.ndarray,
    detected_cars: List,
    box_colors: Tuple,
    filter_points: bool = True,
    visualize: bool = True,
) -> List:
    """
    Performs data association between 2D detections and LiDAR points using frustum culling.
            - List of filtered LiDAR point clusters corresponding to each detection.
    """

    # Perform frustum culling to obtain LiDAR clusters for each bounding box
    point_clusters = frustum_culling(
        lidar,
        [box["bbox_2d"] for box in detected_cars],
        calib,
        image=image,
        filter_points=filter_points,
    )

    filter_point_clusters = remove_outliers_from_clusters(
        point_clusters, nb_neighbors=20, std_ratio=2.0
    )

    if visualize:
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        # Visualize bounding boxes on the image
        draw_boxes(
            image,
            boxes=[box["bbox_2d"] for box in detected_cars],
            box_colors=box_colors,
        )

        # Add LiDAR points for each bounding box cluster
        for i in range(len(filter_point_clusters)):
            pts = filter_point_clusters[i]
            if pts.shape[0] > 0:
                pcd_box = o3d.geometry.PointCloud()
                pcd_box.points = o3d.utility.Vector3dVector(pts[:, :3])
                pcd_box.paint_uniform_color(box_colors[i])  # Color points in red
                pcd += pcd_box

        # Add a coordinate frame for spatial reference
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )

        # Visualize 3D point cloud clusters with coordinate axes
        o3d.visualization.draw_geometries(  # type: ignore
            [pcd, axis], window_name="Frustum Culling Result"
        )

    return filter_point_clusters


def convert_type(obj: Union[np.ndarray, np.generic, Dict, List]) -> Union[List, Dict]:
    """
    Recursively convert NumPy data structures into native Python types to print an output(metadata) in a proper format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_type(item) for item in obj]
    return obj


def compute_bounding_box(
    points: np.ndarray,
    method: str = "OBB",  # Choose: "OBB" or "AABB"
):
    """
    Compute either an Oriented Bounding Box (OBB) using PCA
    or an Axis-Aligned Bounding Box (AABB). No fallbacks.
    """
    pts = np.asarray(points[:, :3], dtype=np.float64)
    if pts.shape[0] == 0:
        return None, -1

    method = method.upper()
    if method not in ["OBB", "AABB"]:
        raise ValueError(f"Invalid method '{method}'. Use 'OBB' or 'AABB'.")

    # Axis-Aligned Bounding Box (AABB)
    if method == "AABB":
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pts)
        )

        _, _, yaw = 0, 0, 0  # No rotation for AABB
        return aabb, yaw

    else:
        # Oriented Bounding Box (OBB) using PCA
        if BOX_USE_MINIMAL: # Use minimal points to construct box
            obb = o3d.geometry.OrientedBoundingBox.create_from_points_minimal(
            o3d.utility.Vector3dVector(pts)
        )
        else:
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pts)
        )
        _, _, yaw = Rot.from_matrix(obb.R).as_euler("xyz", degrees=False)

        return obb, yaw


def main(args=None) -> None:

    parser = argparse.ArgumentParser(description="Load and display KITTI data.")
    parser.add_argument(
        "--idx", type=str, default="000007", help="Index of the KITTI data to load"
    )
    args = parser.parse_args()
    index = args.idx.zfill(6)  

    image, calib, lidar, detected_cars = load_data(index, visualize=True)

    # Assign distinct colors for each detected car
    color_map = plt.colormaps["tab20"]  # type: ignore
    box_colors = tuple(color_map(i)[:3] for i in range(len(detected_cars)))

    # Perform LiDAR-Image data association and visualize results
    point_clusters = perform_data_association(  # noqa: F841
        image, calib, lidar, detected_cars, box_colors, visualize=True
    )


if __name__ == "__main__":
    main()