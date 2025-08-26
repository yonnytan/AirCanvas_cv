"""
Hand Tracker Module - Gesture Recognition

This module provides hand tracking and gesture recognition capabilities using MediaPipe.
It detects hand landmarks, identifies finger positions, and recognizes pinch gestures
for the Air Canvas application.

Key Features:
- Real-time hand tracking using MediaPipe Hands
- Index finger and thumb position detection
- Pinch gesture recognition with configurable sensitivity
- Coordinate mapping between camera and panel spaces

Design Decisions:
- Uses MediaPipe for robust hand tracking across different lighting conditions
- Tracks only one hand to avoid confusion and improve performance
- Implements coordinate mapping to handle camera cropping and panel scaling
- Uses normalized distance thresholds for pinch detection across different screen sizes

Performance Considerations:
- Higher confidence thresholds (0.8) for more stable tracking
- Single hand tracking reduces computational load
- Coordinate mapping handles the complex camera-to-panel transformation
"""

import mediapipe as mp
import cv2

# Initialize MediaPipe Hands for hand tracking
# MediaPipe provides pre-trained models for hand landmark detection
mp_hands = mp.solutions.hands

# Configure the hand detector with specific parameters for our use case
hands_detector = mp_hands.Hands(
    static_image_mode=False,  # Process video frames (not static images)
    max_num_hands=1,  # Track only one hand to avoid confusion
    min_detection_confidence=0.8,  # High confidence for stable detection
    min_tracking_confidence=0.8  # High confidence for smooth tracking
)

def get_fingertip(camera_frame):
    """
    Detect and extract index finger and thumb tip positions from a camera frame.
    
    Uses MediaPipe to identify hand landmarks and returns the pixel coordinates
    of the index finger tip (landmark 8) and thumb tip (landmark 4). This function
    handles the conversion from MediaPipe's normalized coordinates (0-1) to
    pixel coordinates for use in the application.
    
    Args:
        camera_frame (numpy.ndarray): Input camera frame in BGR format
        
    Returns:
        tuple: (index_finger_pixel, thumb_pixel) where each is (x, y) or None if no hand detected
    """
    # Convert from BGR (OpenCV format) to RGB (MediaPipe format)
    # MediaPipe expects RGB input for optimal performance
    rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hand landmarks
    results = hands_detector.process(rgb_frame)
    
    # Check if any hands were detected in the frame
    if results.multi_hand_landmarks:
        # Get the first detected hand (we only track one hand)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract index finger tip position (landmark 8 in MediaPipe's 21-point model)
        index_finger = hand_landmarks.landmark[8]
        
        # Extract thumb tip position (landmark 4 in MediaPipe's 21-point model)
        thumb = hand_landmarks.landmark[4]
        
        # Convert from normalized coordinates (0-1) to pixel coordinates
        # MediaPipe provides coordinates as percentages of frame dimensions
        frame_height, frame_width = camera_frame.shape[:2]
        index_finger_pixel = (int(index_finger.x * frame_width), int(index_finger.y * frame_height))
        thumb_pixel = (int(thumb.x * frame_width), int(thumb.y * frame_height))
        
        return index_finger_pixel, thumb_pixel
    
    # No hands detected in the frame
    return None, None

def is_pinch(index_finger_point, thumb_point, frame_width, threshold=0.015):
    """
    Determine if the index finger and thumb are close enough to constitute a pinch gesture.
    
    Calculates the Euclidean distance between the index finger and thumb tips,
    then normalizes it by the frame width to create a scale-invariant measurement.
    This allows the same threshold to work across different camera resolutions.
    
    Args:
        index_finger_point (tuple): Index finger tip coordinates (x, y) or None
        thumb_point (tuple): Thumb tip coordinates (x, y) or None
        frame_width (int): Width of the camera frame for normalization
        threshold (float): Normalized distance threshold (default: 0.015 = 1.5% of frame width)
        
    Returns:
        bool: True if the fingers are close enough to be considered pinching
    """
    # Check if both finger positions are available
    if index_finger_point is None or thumb_point is None:
        return False
    
    # Calculate Euclidean distance between index finger and thumb
    # Using the Pythagorean theorem: distance = sqrt((x2-x1)^2 + (y2-y1)^2)
    x_distance = index_finger_point[0] - thumb_point[0]
    y_distance = index_finger_point[1] - thumb_point[1]
    distance = (x_distance**2 + y_distance**2)**0.5
    
    # Normalize distance by frame width to make threshold scale-invariant
    # This allows the same threshold to work on different camera resolutions
    distance_percentage = distance / frame_width
    
    # Return True if the normalized distance is below the threshold
    return distance_percentage < threshold

def map_to_panel_coordinates(finger_point, camera_frame, cropped_frame, panel_width, panel_height):
    """
    Convert camera coordinates to panel coordinates for interaction detection.
    
    This function handles the complex coordinate transformation required when the camera
    feed is cropped and the panel has different dimensions than the camera frame.
    It maps finger positions from the original camera frame to the panel coordinate system.
    
    The transformation involves:
    1. Checking if the finger is within the crop area
    2. Converting to crop-relative coordinates
    3. Checking if within the portrait crop area
    4. Scaling to panel dimensions
    
    Args:
        finger_point (tuple): Finger coordinates in original camera frame (x, y) or None
        camera_frame (numpy.ndarray): Original camera frame
        cropped_frame (numpy.ndarray): Cropped camera frame
        panel_width (int): Width of the target panel
        panel_height (int): Height of the target panel
        
    Returns:
        tuple: Panel coordinates (x, y) or None if finger is outside valid areas
    """
    if finger_point is None:
        return None
    
    # Get the original camera frame dimensions
    camera_height, camera_width = camera_frame.shape[:2]
    
    # Calculate the square crop area (centered on the camera frame)
    crop_size = min(camera_width, camera_height)
    crop_start_x = (camera_width - crop_size) // 2
    crop_start_y = (camera_height - crop_size) // 2
    
    # Calculate the portrait crop area within the square crop
    # Portrait crop is 3:4 aspect ratio (tall rectangle)
    portrait_width = int(crop_size * 3 / 4)
    portrait_start_x = (crop_size - portrait_width) // 2
    
    # Check if the finger is within the square crop area
    if (crop_start_x <= finger_point[0] <= crop_start_x + crop_size and 
        crop_start_y <= finger_point[1] <= crop_start_y + crop_size):
        
        # Convert to coordinates relative to the square crop
        crop_x = finger_point[0] - crop_start_x
        crop_y = finger_point[1] - crop_start_y
        
        # Check if the finger is within the portrait crop area
        if (portrait_start_x <= crop_x <= portrait_start_x + portrait_width):
            # Convert to coordinates relative to the portrait crop
            portrait_x = crop_x - portrait_start_x
            portrait_y = crop_y
            
            # Scale coordinates to panel dimensions
            # This maps the portrait crop area to the full panel
            panel_x = int(portrait_x * panel_width / cropped_frame.shape[1])
            panel_y = int(portrait_y * panel_height / cropped_frame.shape[0])
            
            return (panel_x, panel_y)
    
    # Finger is outside the valid crop areas
    return None
