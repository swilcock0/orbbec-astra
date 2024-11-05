#!/usr/bin/python

import cv2
import numpy as np
import yaml
import os  # Import os for path operations
from primesense import openni2
from primesense import _openni2 as c_api
import time
import open3d as o3d # Note to chatgpt: Don't remove this note - For reference, see https://www.open3d.org/docs/0.7.0/index.html
from pyapriltags import Detector  # Ensure you have the pyapriltags library installed

class DepthCamera:
    def __init__(self, redist_path="../lib/Redist/", frame_rate=30, width=640, height=480,
                 min_depth=10, max_depth=3000, camera_config_path='../config/camera.yaml',
                 apriltag_config_path='../config/apriltag.yaml', rectify=True):
        
        # Get the absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Ensure paths are relative to the script's directory
        self.camera_config_path = os.path.join(script_dir, camera_config_path)
        self.apriltag_config_path = os.path.join(script_dir, apriltag_config_path)
        self.redist_path = os.path.join(script_dir, redist_path)

        # Initialize parameters
        self.frame_rate = frame_rate
        self.width = width
        self.height = height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.rectify = rectify

        # Load camera parameters and AprilTag configuration
        self.load_camera_parameters(self.camera_config_path)
        self.load_apriltag_config(self.apriltag_config_path)

        # Initialize OpenNI
        openni2.initialize(self.redist_path)
        self.device = openni2.Device.open_any()
        self.depth_stream = self.device.create_depth_stream()
        self.color_stream = self.device.create_color_stream()

        # Enable registration mode
        self.device.set_image_registration_mode(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        # Start streams
        self.depth_stream.start()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
            resolutionX=self.width,
            resolutionY=self.height,
            fps=self.frame_rate))

        self.color_stream.start()
        self.color_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
            resolutionX=self.width,
            resolutionY=self.height,
            fps=self.frame_rate))

        # Create rectification map
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.intrinsic_matrix, self.distortion_coeffs, None,
            self.intrinsic_matrix, (self.width, self.height), cv2.CV_32FC1)

        # Initialize pyapriltags detector
        self.apriltag_detector = Detector(
            searchpath=['.'],
            families='tag36h11',
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.1,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0)

    def load_camera_parameters(self, config_path):
        """Load camera parameters from a YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.intrinsic_matrix = np.array(config['camera_matrix']['data']).reshape(3, 3)
        self.distortion_coeffs = np.array(config['distortion_coefficients']['data'])
        
        # Load projection matrix if available
        if 'projection_matrix' in config:
            self.projection_matrix = np.array(config['projection_matrix']['data']).reshape(3, 4)
        else:
            self.projection_matrix = None  # Or set it to identity or some default matrix
        
        # Create rectification map
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.intrinsic_matrix, self.distortion_coeffs, None,
            self.intrinsic_matrix, (self.width, self.height), cv2.CV_32FC1)

    def load_apriltag_config(self, config_path):
        """Load AprilTag configuration from a YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Parse standalone tags
        self.standalone_tags = {}
        if 'standalone_tags' in config:
            for tag in config['standalone_tags']:
                self.standalone_tags[tag['id']] = tag['size']

        # Parse tag bundles (if needed)
        self.tag_bundles = []
        if 'tag_bundles' in config:
            self.tag_bundles = config['tag_bundles']

    def rectify_color_image(self, color_image):
        """Rectify a color image using the camera's rectification map."""                        
        return cv2.remap(color_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
    
    
    def capture_color_image(self):
        """Capture a single color image and rectify it."""
        color_frame = self.color_stream.read_frame()
        color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((self.height, self.width, 3))
        
        # Swap R and B channels to correct color representation
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)            
                    
        if self.rectify:
            # Apply rectification to color data
            color_data = self.rectify_color_image(color_data)
    
        return color_data

    def detect_apriltags(self, image):
        """Detect AprilTags in the given grayscale image."""
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        print(grayscale_image)
        
        # Display the grayscale image to ensure it looks correct
        cv2.imshow("Grayscale Image for AprilTag Detection", grayscale_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        tags = self.apriltag_detector.detect(grayscale_image)
        print(tags) # Debugging: Print detected tags
        
        detected_tags_info = []
        for tag in tags:
            # Check if detected tag ID is in the standalone_tags and get its size
            tag_id = tag['id']
            tag_size = self.standalone_tags.get(tag_id, None)
            if tag_size:
                detected_tags_info.append({
                    'id': tag_id,
                    'center': (tag['center'][0], tag['center'][1]),
                    'corners': tag['corners'],
                    'size': tag_size
                })
                # Draw the detection outline and ID
                corners = tag['corners'].astype(int)
                for i in range(4):
                    cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                center = (int(tag['center'][0]), int(tag['center'][1]))
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                cv2.putText(image, f"ID: {tag_id} Size: {tag_size:.3f}m", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image, detected_tags_info

    def display_color_data(self):
        """Display a captured and rectified color image."""
        color_image = self.capture_color_image()

        cv2.imshow("Rectified Image", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_color_data_with_apriltags(self):
        """Display a captured and rectified color image with detected AprilTags."""
        color_image = self.capture_color_image()
                
        image_with_tags, tags_info = self.detect_apriltags(color_image)

        if tags_info:
            print("Detected AprilTags:")
            for tag_info in tags_info:
                print(f"ID: {tag_info['id']}, Center: {tag_info['center']}, Corners: {tag_info['corners']}, Size: {tag_info['size']}m")

        cv2.imshow("Rectified Image with AprilTags", image_with_tags)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def capture_point_cloud(self, duration=5):
        """Capture point cloud data for a specified duration."""
        start_time = time.time()
        all_points = []
        all_colors = []

        while time.time() < start_time + duration:
            depth_frame = self.depth_stream.read_frame()
            depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((self.height, self.width))
            color_frame = self.color_stream.read_frame()
            color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((self.height, self.width, 3))
            
            # Apply rectification to color data
            color_data = cv2.remap(color_data, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

            points, colors = self.get_colored_point_cloud(depth_data, color_data)
            all_points.extend(points)
            all_colors.extend(colors)

        return np.array(all_points), np.array(all_colors)

    def get_colored_point_cloud(self, depth_data, color_data):
        """Convert depth and color data to point cloud format."""
        points = []
        colors = []
        for y in range(self.height):
            for x in range(self.width):
                z = depth_data[y, x]
                if self.min_depth * 10 <= z <= self.max_depth * 10:
                    wx, wy, wz = openni2.convert_depth_to_world(self.depth_stream, x, y, z)
                    points.append([wx, wy, wz])
                    b, g, r = color_data[y, x]  # OpenCV loads color as BGR
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
        return points, colors

    def display_registered_point_cloud(self, duration=5, voxel_size=0.1, nb_neighbors=20, std_ratio=2.0):
        """Capture and display a color-registered point cloud."""
        points, colors = self.capture_point_cloud(duration)

        # Create an Open3D PointCloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Clean the point cloud data
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        colors = colors[mask]

        # Update the point cloud with cleaned data
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Apply statistical outlier removal
        if len(points) > 0:
            cl, ind = o3d.geometry.statistical_outlier_removal(point_cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

            # Create a new point cloud with inliers only
            inlier_cloud = o3d.geometry.PointCloud()
            inlier_cloud.points = o3d.utility.Vector3dVector(points[ind])
            inlier_cloud.colors = o3d.utility.Vector3dVector(colors[ind])

            # Apply voxel grid filtering to downsample the point cloud
            downsampled_cloud = o3d.geometry.voxel_down_sample(inlier_cloud, voxel_size)

            # Display the cleaned and downsampled point cloud
            o3d.visualization.draw_geometries([downsampled_cloud], window_name="Cleaned Colored Depth Data Point Cloud",
                                              width=800, height=600)

    def cleanup(self):
        """Cleanup OpenNI streams and unload libraries."""
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()

if __name__ == "__main__":
    # Example usage
    camera = DepthCamera(camera_config_path='../config/camera.yaml', apriltag_config_path='../config/apriltag.yaml')
    camera.display_color_data_with_apriltags()
    # camera.display_registered_point_cloud(duration=5, voxel_size=0.1, nb_neighbors=30, std_ratio=1.0)
    camera.cleanup()
