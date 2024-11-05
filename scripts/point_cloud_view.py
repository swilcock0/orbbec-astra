#!/usr/bin/python

# ===================================================
# ================ USER PARAMETERS ==================
# ===================================================

# ======= Streaming Parameters ========
frameRate = 30  # Frames per second
width = 640  # Width of image
height = 480  # Height of image

# ======= Computer Vision Parameters =======
minDepth = 10  # Minimum depth (mm)
maxDepth = 3000  # Maximum depth (mm)

# ======= OPENNI2 Paths =======
redistPath = "../lib/Redist/"

# ===================================================
# ================= IMPLEMENTATION ==================
# ===================================================

# Import modules
import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import time
import open3d as o3d

# Initialize OpenNI with its libraries
openni2.initialize(redistPath)

# Open a device
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

# Enable registration mode to align depth and color data
dev.set_image_registration_mode(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)

# Start the depth and color streams with appropriate video modes
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=width, resolutionY=height, fps=frameRate))
color_stream.start()
color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=width, resolutionY=height, fps=frameRate))

# Function to capture and convert depth and color data to point cloud format with color
def get_colored_point_cloud(depth_data, color_data):
    points = []
    colors = []
    for y in range(height):
        for x in range(width):
            z = depth_data[y, x]
            if minDepth * 10 <= z <= maxDepth * 10:  # Apply depth threshold in 100 micrometers
                # Get the 3D world coordinates
                wx, wy, wz = openni2.convert_depth_to_world(depth_stream, x, y, z)
                points.append([wx, wy, wz])
                
                # Get the color for this point and normalize to [0, 1]
                b, g, r = color_data[y, x]  # OpenCV loads color as BGR
                colors.append([r / 255.0, g / 255.0, b / 255.0])
    return points, colors

# Set the end time for 5 seconds
start_time = time.time()
end_time = start_time + 5

# Collect depth and color data for the first 5 seconds
all_points = []
all_colors = []
while time.time() < end_time:
    # Get the depth pixels
    depth_frame = depth_stream.read_frame()
    depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((height, width))

    # Get the color pixels
    color_frame = color_stream.read_frame()
    color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((height, width, 3))
    
    # Add the points and colors from this frame to the point cloud
    points, colors = get_colored_point_cloud(depth_data, color_data)
    all_points.extend(points)
    all_colors.extend(colors)

# Convert the collected points and colors to numpy arrays for Open3D
point_cloud_array = np.array(all_points)
color_array = np.array(all_colors)

# Create an Open3D PointCloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(point_cloud_array)
point_cloud.colors = o3d.utility.Vector3dVector(color_array)

# Set up the Open3D visualization
o3d.visualization.draw_geometries([point_cloud], window_name="Colored Depth Data Point Cloud", width=800, height=600)

# Cleanup
depth_stream.stop()
color_stream.stop()
openni2.unload()
