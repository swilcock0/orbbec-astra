#!/usr/bin/python

import cv2
import numpy as np
from pyapriltags import Detector  # Ensure you have installed pyapriltags

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check the image path.")
        return

    # Convert to grayscale as required by the detector
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image to verify it looks correct
    cv2.imshow("Grayscale Image for AprilTag Detection", grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Initialize the pyapriltags detector
    detector = Detector(
        families='tag36h11',  # Adjust to your tag family
        nthreads=4,
        quad_decimate=0.0,    # Adjust based on your image resolution for speed vs. accuracy
        quad_sigma=3,       # Gaussian blur for noise reduction
        refine_edges=1,
        decode_sharpening=0,
        debug=0
    )

    # Detect AprilTags
    tags = detector.detect(grayscale_image)

    # Print detection results
    if not tags:
        print("No AprilTags detected.")
    else:
        print(f"Detected {len(tags)} AprilTags:")
        for tag in tags:
            print(f"ID: {tag.tag_id}, Center: {tag.center}, Corners: {tag.corners}")

    # Draw detected tags on the original image
    for tag in tags:
        corners = np.array(tag.corners, dtype=int)
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        center = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        cv2.putText(image, f"ID: {tag.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with detected tags
    cv2.imshow("Image with Detected AprilTags", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with the path to your test image
    image_path = 'test2.jpg'
    main(image_path)
