#!/usr/bin/env python3

# --- std ---
import sys
import argparse
from typing import Dict, Tuple

# --- installed ---
import numpy as np
import matplotlib.pyplot as plt

from dataloader import load_data


def project_lidar_to_image(
    lidar_points: np.ndarray,
    calib_dict: Dict,
    img_rgb: np.ndarray,
    filter_points: bool = True,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects LiDAR points to the image plane using calibration data.
    """

    P2 = calib_dict["P2"]
    R0_rect = calib_dict["R0_rect"]
    Tr_velo_to_cam = calib_dict["Tr_velo_to_cam"]

    # Extend R0_rect to a 4x4 homogeneous matrix
    r_h_0_rect_hom = np.eye(4)
    r_h_0_rect_hom[:3, :3] = R0_rect

    # Extend Tr_velo_to_cam to a 4x4 homogeneous matrix
    Tr_h_velo_to_cam_hom = np.eye(4)
    Tr_h_velo_to_cam_hom[:3, :4] = Tr_velo_to_cam

    # Convert LiDAR points to homogeneous coordinates (4xN)
    lidar_points_hom = np.hstack(
        (lidar_points[:, :3], np.ones((lidar_points.shape[0], 1)))
    ).T  # 4xN

    # Transform LiDAR points from Velodyne to camera coordinate frame
    x_h_cam = Tr_h_velo_to_cam_hom @ lidar_points_hom  # 4xN

    # Apply rectification to align with the image plane
    x_h_rect = r_h_0_rect_hom @ x_h_cam  # 4xN

    # Extract depth
    z_cam = x_h_rect[2, :]

    # Depth Filter
    filter_mask = z_cam > 0 if filter_points else np.ones_like(z_cam, dtype=bool)

    # Project rectified 3D points into the image plane
    Y = P2 @ x_h_rect  # 3xN

    # Normalize homogeneous coordinates to obtain pixel (u, v)
    pixel_coords = Y[:2, :] / Y[2, :]  # 2xN
    pixel_coords = pixel_coords.T  # Nx2

    img_height, img_width = img_rgb.shape[:2]  # KITTI image dimensions

    # Filter out points projected outside image boundaries
    image_filter = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 0] <= img_width)
        & (pixel_coords[:, 1] <= img_height)
    )

    # Combine depth and image-bound filters
    combined_mask = filter_mask & image_filter

    # Visualize the projected LiDAR points on the image
    if visualize:
        plt.figure(figsize=(15, 8))
        plt.imshow(img_rgb)
        plt.scatter(
            pixel_coords[combined_mask, 0],
            pixel_coords[combined_mask, 1],
            c=z_cam[combined_mask],
            s=1,
            cmap="jet",
        )
        plt.colorbar(label="Depth (m)")
        plt.axis("off")
        plt.show()

    return pixel_coords, z_cam, combined_mask


def main(args=None) -> None:

    parser = argparse.ArgumentParser(description="Load and display KITTI data.")
    parser.add_argument(
        "--idx",
        type=str,
        default="000044",
        required=True,
        help="Index of the KITTI data to load",
    )
    args = parser.parse_args()
    index = args.idx.zfill(6)  

    # Load KITTI image, calibration data, and LiDAR point cloud
    img_rgb, calib_dict, lidar_arr, _ = load_data(index, visualize=False)

    # Project LiDAR points to the camera image and visualize the result
    projected_points, depths, mask = project_lidar_to_image(
        lidar_arr,
        calib_dict,
        img_rgb,
        visualize=True,
    )

if __name__ == "__main__":
    main()