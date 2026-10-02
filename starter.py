import numpy as np
import cv2

def load_kitti_image(path):
    # OpenCV loads images in BGR format by default
    bgr_image = cv2.imread(path)
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def load_kitti_lidar_scan(path):
    # LiDAR points are stored as a flat array of floats
    scan = np.fromfile(path, dtype=np.float32)
    # Reshape the array to Nx4, where N is the number of points
    points = scan.reshape((-1, 4))
    return points

def load_kitti_calibration(path):
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                # Convert the string of numbers into a NumPy array
                calib[key] = np.array([float(x) for x in value.strip().split()])

    # Reshape matrices to their correct dimensions
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    return calib
