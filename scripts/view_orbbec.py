#!/usr/bin/python

# ===================================================
# ================ USER PARAMETERS ==================
# ===================================================

# ======= Debugging ========
bViewFeed = True  # View the camera feed and blob detection

# ======= Streaming Parameters ========
frameRate = 30  # Frames per second
width = 640  # Width of image
height = 480  # Height of image

# ======= Computer Vision Parameters =======
minIntensity = 200  # Intensity Threshold [0, 255]
minArea = 100  # Minimum blob area in pixels
maxArea = width * height  # Maximum blob area in pixels
minCircularity = 0.5  # Circularity Threshold [0, 1]
minDepth = 10  # Depth Threshold (mm)
maxDepth = 3000
edgePixels = 35  # Edge pixels to ignore in color image
bZeroMask = True  # Remove color pixels without depth
nErosions = 3  # Erosions to apply for mask

# Camera Exposure
bAutoExposure = False
exposure = 100

# ======= Smoothing Parameters =======
easingParam = 0.3  # Smoothing parameter [0 to 0.9]

# ======= OSC UDP Parameters =======
ipAddress = '127.0.0.1'
port = 8888
header = "/orbbec"

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
from pythonosc import osc_message_builder
from pythonosc import udp_client
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Initialize OpenNI with its libraries
openni2.initialize(redistPath)

# Open a device
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=width, resolutionY=height, fps=frameRate))

color_stream = dev.create_color_stream()
color_stream.start()
color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=width, resolutionY=height, fps=frameRate))

dev.set_image_registration_mode(c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
if color_stream.camera:
    color_stream.camera.set_auto_exposure(bAutoExposure)
    color_stream.camera.set_exposure(exposure)
time.sleep(1)

# Configure blob detection
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = minIntensity
params.maxThreshold = 255
params.filterByArea = True
params.minArea = minArea
params.maxArea = maxArea
params.filterByCircularity = True
params.minCircularity = minCircularity
detector = cv2.SimpleBlobDetector_create(params)

# Setup OSC client
client = udp_client.SimpleUDPClient(ipAddress, port)

# Setup for pyqtgraph point cloud visualization
app = pg.mkQApp("Point Cloud Viewer")
window = gl.GLViewWidget()
window.setWindowTitle('Depth Data Point Cloud')
window.setCameraPosition(distance=1000, elevation=20, azimuth=90)
window.show()

# Point cloud data (initially empty)
point_cloud = gl.GLScatterPlotItem()
window.addItem(point_cloud)

# Function to update the point cloud visualization
def update_point_cloud(depth_data):
    points = []
    for y in range(height):
        for x in range(width):
            z = depth_data[y, x]
            if minDepth * 10 <= z <= maxDepth * 10:  # Apply depth threshold in 100 micrometers
                wx, wy, wz = openni2.convert_depth_to_world(depth_stream, x, y, z)
                points.append((wx, wy, wz))
    if points:
        points_array = np.array(points)
        point_cloud.setData(pos=points_array, color=(1, 1, 1, 0.5), size=1)

# Main loop
lastUpdateTime = time.time()
px, py, pz = 0, 0, 0
bPrevData = False

while True:
    # Get the depth pixels
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depthPix = np.frombuffer(frame_data, dtype=np.uint16).reshape((height, width))

    # Update the point cloud visualization
    update_point_cloud(depthPix)

    # Get color pixels and process
    frame = color_stream.read_frame()
    frame_data = frame.get_buffer_as_uint8()
    colorPix = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
    colorPix = np.flip(colorPix, 2)

    grayPix = cv2.cvtColor(colorPix, cv2.COLOR_RGB2GRAY)
    grayPix = (np.multiply(grayPix, (depthPix > minDepth * 10) & (depthPix < maxDepth * 10))).astype(np.uint8)
    _, thresh = cv2.threshold(grayPix, minIntensity, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [x for x in contours if minArea <= cv2.contourArea(x) <= maxArea]
    
    # Select largest contour if it exists
    if contours:
        contour = max(contours, key=cv2.contourArea)
        contourMask = np.zeros_like(grayPix)
        cv2.drawContours(contourMask, [contour], -1, 1, -1)
        validDepthMask = (depthPix != 0).astype(np.uint8)
        depthPix = np.multiply(depthPix, contourMask * validDepthMask)

        # Calculate average depth of contour area
        avgDist = np.sum(depthPix) / np.sum(contourMask) if np.sum(contourMask) > 0 else 0
        if avgDist > 0:
            M = cv2.moments(contour)
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            x, y, z = openni2.convert_depth_to_world(depth_stream, cx, cy, avgDist)

            # Apply smoothing
            if not bPrevData:
                bPrevData = True
                px, py, pz = x, y, z
            x = x * (1 - easingParam) + px * easingParam
            y = y * (1 - easingParam) + py * easingParam
            z = z * (1 - easingParam) + pz * easingParam

            # OSC message with new position
            builder = osc_message_builder.OscMessageBuilder(address=header)
            builder.add_arg(int(True))
            builder.add_arg(x)
            builder.add_arg(y)
            builder.add_arg(z)
            msg = builder.build()
            client.send(msg)

            px, py, pz = x, y, z

    # Debugging the image feed
    if bViewFeed:
        debug_image = cv2.cvtColor(grayPix, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 1)
        cv2.imshow("Image", debug_image)
        cv2.waitKey(34)

    # Update pyqtgraph visualization
    app.processEvents()

# Cleanup
openni2.unload()
cv2.destroyAllWindows()
