#!/usr/bin/python

# Don't remove these comments on import references/websites
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
                 apriltag_config_path='../config/apriltag.yaml', rectify=False, debug_tag_info=False):
        
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
            families='tag36h11',  # Adjust to your tag family
            nthreads=4,
            quad_decimate=0,    # Adjust based on your image resolution for speed vs. accuracy
            quad_sigma=0.1,       # Gaussian blur for noise reduction
            refine_edges=0,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Set debug tag info flag
        self.debug_tag_info = debug_tag_info

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
        
        # Flip the image horizontally to correct the left-to-right mirroring
        color_data = cv2.flip(color_data, 1)    
                    
        if self.rectify:
            # Apply rectification to color data
            color_data = self.rectify_color_image(color_data)
    
        return color_data
    
    def convert_from_3d_to_2d(self, point):
        """Convert a 3D point to a 2D pixel coordinate using the camera's projection matrix."""
        point = np.append(point, 1) # Append 1 to the point to make it homogeneous
        pixel = self.projection_matrix @ point # Apply projection matrix to get pixel coordinates
        pixel = pixel / pixel[2] # Normalize by the third coordinate
        return pixel[:2]

    def estimate_bundle_positions(self, tags):
        bundle_info = []
        for bundle in self.tag_bundles:
            object_points = []
            image_points = []
            
            # For each tag in the bundle layout, gather corners
            for tag_layout in bundle['layout']:
                tag_id = tag_layout['id']
                if tag_id in tags:
                    # Tag's 2D corners detected in the image
                    detected_corners = tags[tag_id]['corners']
                    image_points.extend(detected_corners)
                    
                    # Tag's 3D corner positions in bundle coordinates
                    tag_size = self.standalone_tags[tag_id]
                    half_size = tag_size / 2
                    tag_offset = np.array([tag_layout['x'], tag_layout['y'], tag_layout['z']])

                    # Assuming square tags, construct 3D corners in local bundle coordinates
                    local_corners = np.array([
                        [-half_size, -half_size, 0],
                        [half_size, -half_size, 0],
                        [half_size, half_size, 0],
                        [-half_size, half_size, 0]
                    ]) + tag_offset

                    object_points.extend(local_corners)

            # SolvePnP if we have enough points
            if object_points and image_points:
                object_points = np.array(object_points, dtype=np.float32)
                image_points = np.array(image_points, dtype=np.float32)
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    object_points, image_points, self.intrinsic_matrix, self.distortion_coeffs
                )

                if success:
                    # Project the 3D center of the bundle to 2D
                    bundle_center_3d = np.array([[0, 0, 0]], dtype=np.float32)
                    bundle_center_2d, _ = cv2.projectPoints(
                        bundle_center_3d, rotation_vec, translation_vec, self.intrinsic_matrix, self.distortion_coeffs
                    )

                    # Add bundle pose and center to bundle_info
                    bundle_info.append({
                        'name': bundle['name'],
                        'center': tuple(bundle_center_2d[0].ravel()),
                        'avg_pose_t': rotation_vec,
                        'avg_pose_R': translation_vec,
                        'num_detected_tags': len(bundle['layout'])
                    })

        return bundle_info

    def detect_apriltags(self, image):
        """Detect AprilTags in the given grayscale image, estimate their poses, draw them on the image, and return detailed info."""
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Camera parameters: fx, fy, cx, cy (loaded from the camera configuration)
        camera_params = [self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1], 
                        self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]]
        
        detected_tags_info = {}
        tags = self.apriltag_detector.detect(grayscale_image, estimate_tag_pose=True, camera_params=camera_params, tag_size=self.standalone_tags)

        for tag in tags:
            tag_id = tag.tag_id
                    
            if tag_id in self.standalone_tags:
                detected_tags_info[tag_id] = {
                    'center': (tag.center[0], tag.center[1]),
                    'pose_t': tag.pose_t,
                    'pose_R': tag.pose_R,
                    'corners': tag.corners  # Include corners or any other attribute you need
                }
                # Draw the tag on the image
                corners = np.array(tag.corners, dtype=int)
                for i in range(4):
                    cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                center = tuple([int(i) for i in self.convert_from_3d_to_2d(tag.pose_t)])
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                cv2.putText(image, f"ID: {tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            if self.debug_tag_info:
                if tag_id not in self.standalone_tags:
                    print(f"Tag ID {tag_id} seen, but not found in the standalone tags configuration!\n")
                else:
                    print(f"Tag: {tag_id}, Center: {detected_tags_info[tag_id]['center']}")
                    print(f"Translation (Pose): {detected_tags_info[tag_id]['pose_t'].flatten()}")
                    print(f"Rotation Matrix (Pose):\n{detected_tags_info[tag_id]['pose_R']}\n")

        # Pass detected_tags_info to estimate_bundle_positions
        bundle_info = self.estimate_bundle_positions(detected_tags_info)
        for bundle in bundle_info:
            bundle_center = tuple(map(int, bundle['center']))
            cv2.circle(image, bundle_center, 7, (255, 255, 0), -1)
            cv2.putText(image, f"Bundle: {bundle['name']}", (bundle_center[0] + 10, bundle_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if self.debug_tag_info:
                print(f"Bundle: {bundle['name']}, Center: {bundle['center']}")
                print(f"Num Detected Tags: {bundle['num_detected_tags']}")
                print(f"Estimated Pose (Translation): {bundle['avg_pose_t']}")
                print(f"Estimated Pose (Rotation):\n{bundle['avg_pose_R']}\n")
        
        if self.debug_tag_info:
            print("-------------------------------------------------------\n")        

        return image, detected_tags_info, bundle_info

    def realtime_apriltag_detection(self):
        """Open a live visualization of the image with overlaid AprilTag detections and bundle info."""
        print("Press 'q' to exit the live AprilTag detection window.")
        try:
            while True:
                color_image = self.capture_color_image()
                image_with_tags, _, _ = self.detect_apriltags(color_image)

                cv2.imshow("Live AprilTag Detection with Tags and Bundles", image_with_tags)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()

    def cleanup(self):
        """Cleanup OpenNI streams and unload libraries."""
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()

if __name__ == "__main__":
    # Example usage
    camera = DepthCamera(camera_config_path='../config/camera.yaml', apriltag_config_path='../config/apriltag.yaml', debug_tag_info=True)
    camera.realtime_apriltag_detection()
    #camera.display_color_data_with_apriltags()
    #camera.display_registered_point_cloud(duration=2, voxel_size=1, nb_neighbors=100, std_ratio=1.0)
    camera.cleanup()
