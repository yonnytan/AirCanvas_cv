"""
Canvas Module - Drawing Surface Management

This module handles the creation and manipulation of drawing canvases for the Air Canvas
application. It provides functions for creating blank canvases, drawing lines, and
overlaying drawings on camera feeds.

Key Features:
- Canvas creation with customizable background colors
- Line drawing with anti-aliasing for smooth appearance
- Canvas overlay system that works with both white and black backgrounds
- Background-aware masking for proper drawing visibility

Design Decisions:
- Uses numpy arrays for efficient image manipulation
- Supports both white and black canvas backgrounds
- Implements proper masking to preserve camera feed where no drawing exists
- Uses OpenCV's anti-aliased line drawing for quality
"""

import cv2
import numpy as np

def create_canvas(height, width, color=(255,255,255)):
    """
    Create a blank canvas for drawing.
    
    Creates a numpy array filled with the specified color to serve as a drawing
    surface. The canvas is created in BGR format (OpenCV standard) with 3 color
    channels and 8-bit depth for compatibility with OpenCV functions.
    
    Args:
        height (int): Canvas height in pixels
        width (int): Canvas width in pixels
        color (tuple): Background color in BGR format (default: white)
        
    Returns:
        numpy.ndarray: Blank canvas as a 3-channel BGR image
    """
    # Create a 3D array with shape (height, width, 3) filled with the specified color
    # dtype=np.uint8 ensures 8-bit color values (0-255) for OpenCV compatibility
    return np.full((height, width, 3), color, dtype=np.uint8)

def clear_canvas(canvas, color=(255,255,255)):
    """
    Clear the canvas by filling it with a specified color.
    
    This function modifies the canvas in-place by setting all pixels to the
    specified color. It's more efficient than creating a new canvas when
    you want to clear existing drawings.
    
    Args:
        canvas (numpy.ndarray): Canvas to clear
        color (tuple): Color to fill the canvas with (BGR format)
    """
    # Fill the entire canvas with the specified color
    # The [:] syntax modifies the array in-place without creating a new copy
    canvas[:] = color

def draw_line(canvas, start_point, end_point, color, thickness):
    """
    Draw a line on the canvas.
    
    Uses OpenCV's line drawing function with anti-aliasing for smooth,
    high-quality lines. The line is drawn directly on the canvas array.
    
    Args:
        canvas (numpy.ndarray): Canvas to draw on
        start_point (tuple): Starting point (x, y) in pixels
        end_point (tuple): Ending point (x, y) in pixels
        color (tuple): Line color in BGR format
        thickness (int): Line thickness in pixels
    """
    # Draw a line with anti-aliasing (cv2.LINE_AA) for smooth appearance
    # The line is drawn directly on the canvas array
    cv2.line(canvas, start_point, end_point, color, thickness, cv2.LINE_AA)

def overlay_canvas(camera_frame, canvas, background_color=(255,255,255)):
    """
    Overlay the drawing canvas on top of the camera frame.
    
    This function combines the camera feed with the drawing canvas using
    intelligent masking. It preserves the camera background where no drawing
    exists and shows the drawing where lines have been drawn.
    
    The masking system adapts to the canvas background color:
    - White canvas: Looks for non-white pixels (drawn content)
    - Black canvas: Looks for non-black pixels (drawn content)
    
    Args:
        camera_frame (numpy.ndarray): Camera feed image
        canvas (numpy.ndarray): Drawing canvas
        background_color (tuple): Canvas background color for masking
        
    Returns:
        numpy.ndarray: Combined image with drawing overlaid on camera feed
    """
    # Resize canvas to match camera frame dimensions for proper overlay
    canvas_resized = cv2.resize(canvas, (camera_frame.shape[1], camera_frame.shape[0]))
    
    # Ensure canvas is in the correct data type for OpenCV operations
    canvas_uint8 = canvas_resized.astype(np.uint8)
    
    # Convert canvas to grayscale for thresholding
    # This allows us to identify drawn vs. background pixels
    gray_canvas = cv2.cvtColor(canvas_uint8, cv2.COLOR_BGR2GRAY)
    
    # Create a mask to identify drawn areas based on background color
    if background_color == (255, 255, 255):  # White background
        # Find pixels that are NOT white (where drawing exists)
        # Threshold at 250 to account for slight color variations
        _, mask = cv2.threshold(gray_canvas, 250, 255, cv2.THRESH_BINARY_INV)
    else:  # Black background
        # Find pixels that are NOT black (where drawing exists)
        # Threshold at 5 to account for slight color variations
        _, mask = cv2.threshold(gray_canvas, 5, 255, cv2.THRESH_BINARY)
    
    # Create inverse mask for background areas (where no drawing exists)
    background_mask = cv2.bitwise_not(mask)
    
    # Apply masks to separate background and foreground
    # Keep camera background where no drawing exists
    background = cv2.bitwise_and(camera_frame, camera_frame, mask=background_mask)
    # Keep canvas drawing where drawing exists
    foreground = cv2.bitwise_and(canvas_uint8, canvas_uint8, mask=mask)
    
    # Combine background and foreground to create final overlay
    result = cv2.add(background, foreground)
    return result
