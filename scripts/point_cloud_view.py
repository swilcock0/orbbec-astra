#!/usr/bin/python

import cv2
import numpy as np
import yaml
import os  # Import os for path operations
from primesense import openni2
from primesense import _openni2 as c_api
import time
import open3d as o3d  # Note to chatgpt: Don't remove this note - For reference, see https://www.open3d.org/docs/0.7.0/index.html
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
        tags = self.apriltag_detector.detect(grayscale_image)
        
        detected_tags_info = []
        tag_centers = {}
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
                tag_centers[tag_id] = (tag['center'][0], tag['center'][1])

                # Draw the detection outline and ID
                corners = tag['corners'].astype(int)
                for i in range(4):
                    cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                center = (int(tag['center'][0]), int(tag['center'][1]))
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                cv2.putText(image, f"ID: {tag_id} Size: {tag_size:.3f}m", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Estimate bundle positions
        bundle_info = self.estimate_bundle_positions(tag_centers)
        for bundle in bundle_info:
            bundle_center = tuple(map(int, bundle['center']))
            cv2.circle(image, bundle_center, 7, (255, 255, 0), -1)
            cv2.putText(image, f"Bundle: {bundle['name']}", bundle_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return image, detected_tags_info, bundle_info

    def estimate_bundle_positions(self, tag_centers):
        """Estimate the center of tag bundles based on detected tags."""
        bundle_info = []
        for bundle in self.tag_bundles:
            detected_tag_positions = []
            for tag_layout in bundle['layout']:
                tag_id = tag_layout['id']
                if tag_id in tag_centers:
                    detected_tag_positions.append(tag_centers[tag_id])

            if detected_tag_positions:
                # Calculate the average position of detected tags for the bundle
                avg_x = np.mean([pos[0] for pos in detected_tag_positions])
                avg_y = np.mean([pos[1] for pos in detected_tag_positions])
                bundle_info.append({
                    'name': bundle['name'],
                    'center': (avg_x, avg_y),
                    'num_detected_tags': len(detected_tag_positions)
                })

        return bundle_info

    def display_color_data_with_apriltags(self):
        """Display a captured and rectified color image with detected AprilTags and bundles."""
        color_image = self.capture_color_image()
        image_with_tags, tags_info, bundle_info = self.detect_apriltags(color_image)

        if tags_info:
            print("Detected AprilTags:")
            for tag_info in tags_info:
                print(f"ID: {tag_info['id']}, Center: {tag_info['center']}, Corners: {tag_info['corners']}, Size: {tag_info['size']}m")

        if bundle_info:
            print("Estimated Bundle Positions:")
            for bundle in bundle_info:
                print(f"Bundle: {bundle['name']}, Center: {bundle['center']}, Number of Detected Tags: {bundle['num_detected_tags']}")

        cv2.imshow("Rectified Image with AprilTags and Bundles", image_with_tags)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Other methods remain unchanged
