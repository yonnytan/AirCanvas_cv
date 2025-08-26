"""
Camera Module - Webcam Interface

This module provides a simplified interface for accessing and reading from webcams.
It handles camera initialization, frame capture, and error handling for the Air Canvas
application.

Design Decisions:
- Uses OpenCV for cross-platform camera access
- Provides error handling for common camera issues
- Abstracts camera complexity from main application
- Uses descriptive variable names for clarity
"""

import cv2

def open_camera(camera_number=0):
    """
    Initialize and open a webcam for video capture.
    
    Most systems have the built-in webcam as camera 0, but external cameras
    may use different numbers. This function handles camera initialization
    and provides clear error messages if the camera cannot be accessed.
    
    Args:
        camera_number (int): Camera device number (0 for default/built-in camera)
        
    Returns:
        cv2.VideoCapture: Open camera object ready for frame capture
        
    Raises:
        RuntimeError: If camera cannot be opened (e.g., camera in use, disconnected)
    """
    # OpenCV's VideoCapture handles camera initialization across platforms
    camera = cv2.VideoCapture(camera_number)
    
    # Check if camera opened successfully - this catches common issues like:
    # - Camera already in use by another application
    # - Camera disconnected or not found
    # - Insufficient permissions
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera - make sure it's connected!")
    
    return camera

def get_frame(camera):
    """
    Capture a single frame from the camera.
    
    This function reads one image from the camera and handles potential
    capture failures. It's designed to be called repeatedly in a loop
    for continuous video processing.
    
    Args:
        camera: Open camera object from open_camera()
        
    Returns:
        numpy.ndarray: BGR image frame as a numpy array
        
    Raises:
        RuntimeError: If frame capture fails (e.g., camera disconnected during use)
    """
    # Read one frame from the camera
    # success indicates if the frame was captured successfully
    # frame contains the actual image data
    success, frame = camera.read()
    
    # Check if frame capture was successful
    # This can fail if the camera is disconnected during use
    # or if there are hardware issues
    if not success:
        raise RuntimeError("Could not get a picture from the camera!")
    
    return frame


