#!/usr/bin/env python3

import sys
import argparse
from typing import Dict, List, Tuple
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from starter import load_kitti_calibration, load_kitti_image, load_kitti_lidar_scan, load_kitti_labels

MIN_IDX = 0
MAX_IDX = 200

def load_data(
    index: str = "000044", visualize: bool = True
) -> Tuple[np.ndarray, Dict, np.ndarray, List[Dict]]:
    """
    Loads and optionally visualizes KITTI dataset components for a given sample index.
    """

    # Index range check (must be between 0 and 200 inclusive)
    idx_int = int(index)
    if idx_int < MIN_IDX or MAX_IDX > 200:
        raise ValueError(f"Index must be between 000000 and 000200 (got {index})")

    # File paths for the selected KITTI dataset
    image_path: str = f"./training/image_2/{index}.png"
    velodyne_path: str = f"./training/velodyne/{index}.bin"
    calibration_path: str = f"./training/calib/{index}.txt"
    label_path: str = f"./training/label_2/{index}.txt"

    # Loading KITTI data
    img_rgb: np.ndarray = load_kitti_image(image_path)
    lidar_arr: np.ndarray = load_kitti_lidar_scan(velodyne_path)
    calib_dict: Dict = load_kitti_calibration(calibration_path)
    label_list: List[Dict] = load_kitti_labels(label_path)

    # Visualization part
    if visualize:
        # Display the image using matplotlib
        plt.figure(figsize=(15, 8))
        plt.imshow(img_rgb)
        plt.title(f"RGB Image [{index}]")
        plt.axis("off")
        plt.show()

        # Visualize the LiDAR points using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_arr[:, :3])
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Set point color to gray
        o3d.visualization.draw_geometries(  # type: ignore
            [pcd], window_name=f"LiDAR Point Cloud [{index}]"
        )

    return img_rgb, calib_dict, lidar_arr, label_list


def main(args=None) -> None:

    parser = argparse.ArgumentParser(description="Load and display KITTI data.")
    parser.add_argument(
        "--idx",
        type=str,
        default="000044",
        required=True,
        help='Index of the KITTI data to load (e.g., "000044")',
    )
    args = parser.parse_args()

    # Load and visualize KITTI sample data
    index = args.idx.zfill(6)  

    img_rgb, calib_dict, lidar_arr, label_list = load_data(index, visualize=True)


if __name__ == "__main__":
    main()