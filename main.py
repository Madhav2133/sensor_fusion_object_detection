#!/usr/bin/env python3

# --- std ---
import sys
import argparse
from pprint import pprint


# --- installed ---
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# --- user ---
from dataloader import load_data
from sensor_fusion import project_lidar_to_image
from data_association import (
    convert_type,
    perform_data_association,
    compute_bounding_box,
)

BOX_TYPE = "AABB"



def main(args=None):

    parser = argparse.ArgumentParser(description="Load and display KITTI data.")
    parser.add_argument(
        "--idx", type=str, default="000007", help="Index of the KITTI data to load"
    )
    args = parser.parse_args()
    index = args.idx.zfill(6)  

    image, calib, lidar, detected_cars = load_data(index, visualize=True)
   
    color_map = plt.colormaps["tab10"]  # type: ignore
    colors = tuple(color_map(i)[:3] for i in range(len(detected_cars)))

    # Perform data association to match LiDAR points with detected objects
    # Returns list of point clusters (one per detected car)
    point_clusters = perform_data_association(
        image, calib, lidar, detected_cars, colors, visualize=True
    )

    # Visualize the LiDAR points and bounding boxes using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Color points in gray

    line_set = o3d.geometry.LineSet()
    new_pcd = list()
    output_list = list()

    # Loop through all detected clusters (cars)
    for i, cluster in enumerate(point_clusters):
        if cluster.shape[0] == 0:
            continue

        # Create colored point cloud for each detected car cluster
        pcd_box = o3d.geometry.PointCloud()
        pcd_box.points = o3d.utility.Vector3dVector(cluster[:, :3])
        pcd_box.paint_uniform_color(colors[i])
        new_pcd.append(pcd_box)

        # get the bounding box
        box, yaw = compute_bounding_box(cluster, method=BOX_TYPE)

        if box is None:
            continue
        
        # get box dimensions
        if BOX_TYPE == "AABB":
            length, width, height = box.get_extent()           
        else:
            length, width, height = box.extent

        # Box metrics and projection
        center = box.get_center()
        corners = np.array(box.get_box_points())

        # Project 3D box corners onto 2D image plane using camera calibration
        projected_points, depths, mask = project_lidar_to_image(
            corners, calib, image, visualize=False, filter_points=False
        )

        # Store metadata for current bounding box
        output = {
            "frame": "camera",
            "type": BOX_TYPE,
            "center_m": center,
            "dims_m": {"l": length, "w": width, "h": height},
            "yaw_rad": yaw,
            "n_points": cluster.shape[0],
        }
        output_list.append(output)

        # Draw 3D box on image
        if BOX_TYPE == "AABB":
            line_set_object = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
        else:
            line_set_object = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        lines = np.asarray(line_set_object.lines)
        line_color = tuple(int(c) for c in (np.array(colors[i][:3]) * 255))

        for line in lines:
            pt1 = projected_points[line[0]]
            pt2 = projected_points[line[1]]
            cv2.line(
                image,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                line_color,
                2,
            )

        # Color box wireframe in 3D visualization
        line_set_object.paint_uniform_color(colors[i])  # Color box lines in red
        line_set += line_set_object

    # Print all computed bounding box information
    print("-" * 50)
    pprint(convert_type(output_list))

    # Visualization
    cv2.imshow("Projected 3D Bounding Boxes", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display 3D point clouds and bounding boxes
    o3d.visualization.draw_geometries(  # type: ignore
        [pcd, line_set] + new_pcd, window_name="3D Bounding Boxes"
    )


if __name__ == "__main__":
    main()