#!/usr/bin/python

# Import necessary modules
import cv2
import numpy as np
import yaml
import os  # Import os for path operations
from primesense import openni2
from primesense import _openni2 as c_api
import time
import open3d as o3d  # https://www.open3d.org/docs/0.7.0/index.html
from pyapriltags import Detector  # https://github.com/WillB97/pyapriltags
from scipy.spatial.transform import Rotation as R

class DepthCamera:
    def __init__(self, redist_path="../lib/Redist/", frame_rate=30, width=640, height=480,
                 min_depth=10, max_depth=3000, camera_config_path='../config/camera.yaml',
                 apriltag_config_path='../config/apriltag.yaml', rectify=False, debug_tag_info=False,
                 smoothing_alpha=0.8):  # Add smoothing_alpha as a parameter
        """
        Initializes the DepthCamera object with various parameters.
        
        Args:
            redist_path (str): Path to Redist library for OpenNI.
            frame_rate (int): Frame rate of the camera.
            width (int): Width of the camera frames.
            height (int): Height of the camera frames.
            min_depth (int): Minimum depth in mm.
            max_depth (int): Maximum depth in mm.
            camera_config_path (str): Path to camera configuration YAML file.
            apriltag_config_path (str): Path to AprilTag configuration YAML file.
            rectify (bool): Whether to rectify images.
            debug_tag_info (bool): Whether to output debug information for tags.
            smoothing_alpha (float): Exponential smoothing factor for poses.
        """
        
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
            quad_decimate=0,
            quad_sigma=0.1,
            refine_edges=0,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Smoothing factor for exponential smoothing
        self.smoothing_alpha = smoothing_alpha

        # Set debug tag info flag
        self.debug_tag_info = debug_tag_info

        # Dictionary to store previous smoothed poses for each tag
        self.filtered_poses = {}

    def load_camera_parameters(self, config_path):
        """Load camera parameters from a YAML file.
        
        Args:
            config_path (str): Path to the YAML file with camera parameters.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.intrinsic_matrix = np.array(config['camera_matrix']['data']).reshape(3, 3)
        self.distortion_coeffs = np.array(config['distortion_coefficients']['data'])
        
        if 'projection_matrix' in config:
            self.projection_matrix = np.array(config['projection_matrix']['data']).reshape(3, 4)
        else:
            self.projection_matrix = None

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.intrinsic_matrix, self.distortion_coeffs, None,
            self.intrinsic_matrix, (self.width, self.height), cv2.CV_32FC1)

    def load_apriltag_config(self, config_path):
        """Load AprilTag configuration from a YAML file.
        
        Args:
            config_path (str): Path to the YAML file with AprilTag configurations.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.standalone_tags = {}
        if 'standalone_tags' in config:
            for tag in config['standalone_tags']:
                self.standalone_tags[tag['id']] = tag['size']

        self.tag_bundles = []
        if 'tag_bundles' in config:
            self.tag_bundles = config['tag_bundles']

    def rectify_color_image(self, color_image):
        """Rectify the color image based on camera parameters.
        
        Args:
            color_image (np.ndarray): Input color image.
        
        Returns:
            np.ndarray: Rectified color image.
        """
        return cv2.remap(color_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def capture_color_image(self):
        """Capture and return a color image from the camera.
        
        Returns:
            np.ndarray: The captured color image.
        """
        color_frame = self.color_stream.read_frame()
        color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((self.height, self.width, 3))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.flip(color_data, 1)
                    
        if self.rectify:
            color_data = self.rectify_color_image(color_data)
    
        return color_data

    def convert_from_3d_to_2d(self, point):
        """Convert a 3D point to a 2D point in image coordinates.
        
        Args:
            point (np.ndarray): The 3D point.
        
        Returns:
            np.ndarray: The 2D point in pixel coordinates.
        """
        point = np.append(point, 1)
        pixel = self.projection_matrix @ point
        pixel = pixel / pixel[2]
        return pixel[:2]

    def exponential_smooth_pose(self, tag_id, new_pose_t, new_pose_R):
        """Smooth the pose of a detected tag using exponential smoothing.
        
        Args:
            tag_id (int): The ID of the AprilTag.
            new_pose_t (np.ndarray): The new translation vector.
            new_pose_R (np.ndarray): The new rotation matrix.
        
        Returns:
            tuple: The smoothed translation and rotation matrices.
        """
        if tag_id not in self.filtered_poses:
            self.filtered_poses[tag_id] = {
                'pose_t': new_pose_t,
                'pose_R': new_pose_R
            }
            return new_pose_t, new_pose_R
        
        prev_pose_t = self.filtered_poses[tag_id]['pose_t']
        prev_pose_R = R.from_matrix(self.filtered_poses[tag_id]['pose_R'])

        smoothed_pose_t = self.smoothing_alpha * new_pose_t + (1 - self.smoothing_alpha) * prev_pose_t

        new_rotation = R.from_matrix(new_pose_R)
        smoothed_rotation = R.slerp(
            R.from_quat([prev_pose_R.as_quat(), new_rotation.as_quat()]),
            np.array([0, self.smoothing_alpha])
        ).as_matrix()

        self.filtered_poses[tag_id]['pose_t'] = smoothed_pose_t
        self.filtered_poses[tag_id]['pose_R'] = smoothed_rotation

        return smoothed_pose_t, smoothed_rotation

    def detect_apriltags(self, image):
        """Detect AprilTags in the given image and apply exponential smoothing to their poses.
        
        Args:
            image (np.ndarray): The input color image.
        
        Returns:
            tuple: The image with drawn tags, a dictionary of detected tag information, and bundle info.
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        camera_params = [self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1], 
                         self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]]
        
        standalone_tag_info = {}
        tag_bundle_info = []

        tags = self.apriltag_detector.detect(grayscale_image, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.065)

        for tag in tags:
            tag_id = tag.tag_id
            corners = tag.corners

            tag_pose_t, tag_pose_R = tag.pose_t, tag.pose_R
            smoothed_pose_t, smoothed_pose_R = self.exponential_smooth_pose(tag_id, tag_pose_t, tag_pose_R)

            standalone_tag_info[tag_id] = {
                'pose_t': smoothed_pose_t,
                'pose_R': smoothed_pose_R
            }

            for i, corner in enumerate(corners):
                corner = tuple(map(int, corner))
                cv2.line(image, corner, tuple(map(int, corners[(i + 1) % 4])), (0, 255, 0), 2)
                cv2.circle(image, corner, 5, (0, 0, 255), -1)

            if self.debug_tag_info:
                print(f"Detected tag ID: {tag_id}, Pose: {smoothed_pose_t}")

        return image, standalone_tag_info, tag_bundle_info
    
    def realtime_apriltag_detection(self):
        """Run a real-time AprilTag detection with point cloud visualization."""
        while True:
            color_image = self.capture_color_image()
            if color_image is None:
                break

            detected_image, standalone_tag_info = self.detect_apriltags(color_image)
            cv2.imshow("AprilTags Detection", detected_image)

            # Capture and visualize the point cloud
            point_cloud = self.capture_point_cloud()
            if point_cloud:
                self.point_cloud_registration(point_cloud)
                self.visualize_point_cloud(point_cloud)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def __del__(self):
        """Clean up OpenNI resources."""
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()

def main():
    """
    Main function to initialize the DepthCamera, capture frames, and perform AprilTag detection.
    """
    camera = DepthCamera(debug_tag_info=True)
    camera.realtime_apriltag_detection()

if __name__ == "__main__":
    main()
