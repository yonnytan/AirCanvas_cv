"""
Controls Module - User Interface Elements

This module handles the creation and interaction with UI controls for the Air Canvas
application. It provides functions for color selection, slider interaction, and
visual feedback for user interactions.

Key Features:
- Color button detection and selection
- Color wheel picker for custom color selection
- Thickness slider rendering and interaction
- Visual feedback for hover states
- Coordinate-based interaction detection

Design Decisions:
- Uses coordinate-based detection for precise interaction
- Implements visual feedback to show current selections
- Provides both predefined colors and custom color wheel control
- Uses vertical slider for thickness adjustment
- Separates UI rendering from interaction logic

Interaction Model:
- Hover-based color selection (no click required)
- Color wheel interaction based on finger position (radius and angle)
- Visual feedback with borders and color changes
- Persistent state management for selected colors
"""

import cv2
import numpy as np
import math

# Global variable to store the pre-generated color wheel
_color_wheel_cache = None

def create_controls_window():
    """
    Create a separate window with sliders for RGB and thickness control.
    
    This function creates a dedicated controls window that can be used as an
    alternative to the in-panel sliders. The window contains trackbars for
    Red, Green, Blue (0-255) and Thickness (5-20) values.
    
    Note: This function is currently unused in the main application as sliders
    are integrated directly into the main window for better user experience.
    """
    # Create a window called "Controls" for the trackbars
    cv2.namedWindow("Controls")
    
    # Create trackbars for color and thickness control
    # Each trackbar has a name, window, initial value, max value, and callback
    # The lambda x: None callback prevents errors when no callback is needed
    cv2.createTrackbar("R", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("G", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("B", "Controls", 0, 255, lambda x: None)
    cv2.createTrackbar("Thickness", "Controls", 5, 20, lambda x: None)

def get_control_values():
    """
    Retrieve current values from the control window trackbars.
    
    This function reads the current position of each trackbar in the controls
    window and returns the values. It's designed to work with the separate
    controls window created by create_controls_window().
    
    Returns:
        tuple: (red_value, green_value, blue_value, thickness_value)
    """
    # Read the current position of each trackbar
    # getTrackbarPos returns the current value of the specified trackbar
    red_value = cv2.getTrackbarPos("R", "Controls")
    green_value = cv2.getTrackbarPos("G", "Controls")
    blue_value = cv2.getTrackbarPos("B", "Controls")
    thickness_value = cv2.getTrackbarPos("Thickness", "Controls")
    
    return red_value, green_value, blue_value, thickness_value

def check_color_selection(finger_point, color_buttons):
    """
    Check if the finger is hovering over a color button and return selection info.
    
    This function determines which color button (if any) the finger is currently
    hovering over. It returns information about the selected button including
    whether it's an eraser and the button's color value.
    
    Args:
        finger_point (tuple): Finger coordinates (x, y) or None if no finger detected
        color_buttons (dict): Dictionary of color button definitions
        
    Returns:
        tuple: (button_name, is_eraser, button_color) or (None, False, None) if no selection
    """
    # If no finger is detected, return no selection
    if finger_point is None:
        return None, False, None
    
    # Check each color button for intersection with finger position
    for button_name, (button_coords, button_color) in color_buttons.items():
        x1, y1, x2, y2 = button_coords
        
        # Check if finger is within this button's rectangular bounds
        if x1 <= finger_point[0] <= x2 and y1 <= finger_point[1] <= y2:
            # Determine if this button is the eraser
            is_eraser = (button_name == "eraser")
            return button_name, is_eraser, button_color
    
    # Finger is not over any button
    return None, False, None

def draw_color_wheel(panel, center_x, center_y, radius, current_color):
    """
    Draw a filled color wheel for custom color selection.
    
    Creates a smooth, filled circular color wheel using HSV color space for intuitive color selection.
    The wheel shows the full spectrum of hues with varying saturation from center to edge.
    The center shows the currently selected color.
    
    Args:
        panel (numpy.ndarray): Panel image to draw the color wheel on
        center_x, center_y (int): Center coordinates of the color wheel
        radius (int): Radius of the color wheel
        current_color (tuple): Current RGB color to display in center (BGR format)
        
    Returns:
        dict: Color wheel coordinates and parameters for interaction detection
    """
    global _color_wheel_cache
    
    # Use cached color wheel if available and same size
    if _color_wheel_cache is None or _color_wheel_cache.shape[0] != radius * 2:
        # Create the color wheel image once and cache it
        wheel_size = radius * 2
        wheel_image = np.zeros((wheel_size, wheel_size, 3), dtype=np.uint8)
        
        # Generate colors for each pixel in the wheel
        for y in range(wheel_size):
            for x in range(wheel_size):
                # Calculate distance from center
                dx = x - radius
                dy = y - radius
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Only process pixels within the circle
                if distance <= radius:
                    # Calculate angle (hue)
                    angle = math.degrees(math.atan2(dy, dx))
                    if angle < 0:
                        angle += 360
                    
                    # Calculate saturation based on distance from center
                    # Outer edge = full saturation, center = no saturation (white)
                    saturation = distance / radius
                    saturation = max(0.1, min(1.0, saturation))  # Clamp between 0.1 and 1.0
                    
                    # Convert to HSV color space
                    # OpenCV uses H: 0-179, S: 0-255, V: 0-255
                    hue = angle / 2  # Convert to OpenCV hue range
                    saturation_cv = int(saturation * 255)
                    value = 255  # Full brightness
                    
                    # Convert HSV to BGR
                    hsv_color = np.array([[[hue, saturation_cv, value]]], dtype=np.uint8)
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                    
                    # Set the pixel color
                    wheel_image[y, x] = bgr_color
        
        # Cache the generated wheel
        _color_wheel_cache = wheel_image
    
    # Use the cached wheel
    wheel_image = _color_wheel_cache
    
    # Calculate the region to copy the wheel to the panel
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(panel.shape[1], center_x + radius)
    y2 = min(panel.shape[0], center_y + radius)
    
    # Calculate the corresponding region in the wheel image
    wheel_x1 = max(0, radius - (center_x - x1))
    wheel_y1 = max(0, radius - (center_y - y1))
    wheel_x2 = min(wheel_image.shape[1], radius + (x2 - center_x))
    wheel_y2 = min(wheel_image.shape[0], radius + (y2 - center_y))
    
    # Copy the wheel to the panel with proper masking
    # This ensures only the circular area is copied, preventing background issues
    wheel_region = wheel_image[wheel_y1:wheel_y2, wheel_x1:wheel_x2]
    
    # Create a mask for the circular area
    region_height, region_width = wheel_region.shape[:2]
    mask = np.zeros((region_height, region_width), dtype=np.uint8)
    
    # Calculate the center and radius for the mask
    mask_center_x = region_width // 2
    mask_center_y = region_height // 2
    mask_radius = min(mask_center_x, mask_center_y)
    
    # Create circular mask
    cv2.circle(mask, (mask_center_x, mask_center_y), mask_radius, 255, -1)
    
    # Apply mask to wheel region
    masked_wheel = cv2.bitwise_and(wheel_region, wheel_region, mask=mask)
    
    # Get the existing panel region
    panel_region = panel[y1:y2, x1:x2].copy()
    
    # Create inverse mask for background
    background_mask = cv2.bitwise_not(mask)
    background_region = cv2.bitwise_and(panel_region, panel_region, mask=background_mask)
    
    # Combine wheel and background
    combined_region = cv2.add(masked_wheel, background_region)
    
    # Copy back to panel
    panel[y1:y2, x1:x2] = combined_region
    
    # Draw inner circle showing current color
    inner_radius = radius // 3
    cv2.circle(panel, (center_x, center_y), inner_radius, current_color, -1)
    
    # Draw border around the color wheel
    cv2.circle(panel, (center_x, center_y), radius, (255, 255, 255), 2)
    
    # Return color wheel parameters for interaction detection
    return {
        'center_x': center_x,
        'center_y': center_y,
        'radius': radius,
        'inner_radius': inner_radius
    }

def check_color_wheel_interaction(finger_point, color_wheel_coords):
    """
    Check if finger is over the color wheel and calculate selected color.
    
    This function detects when the finger is hovering over the color wheel and
    calculates the selected color based on the finger's position (angle and radius).
    The angle determines the hue, and the distance from center determines saturation.
    
    Args:
        finger_point (tuple): Finger coordinates (x, y) or None
        color_wheel_coords (dict): Color wheel parameters from draw_color_wheel()
        
    Returns:
        tuple: (red, green, blue) color values or None if not interacting
    """
    if finger_point is None:
        return None
    
    center_x = color_wheel_coords['center_x']
    center_y = color_wheel_coords['center_y']
    radius = color_wheel_coords['radius']
    inner_radius = color_wheel_coords['inner_radius']
    
    # Calculate distance from finger to center
    dx = finger_point[0] - center_x
    dy = finger_point[1] - center_y
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Check if finger is within the color wheel area
    if distance <= radius and distance >= inner_radius:
        # Calculate angle from center (0-360 degrees)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        
        # Calculate saturation based on distance from center
        # Closer to center = less saturated, closer to edge = more saturated
        saturation = (distance - inner_radius) / (radius - inner_radius)
        saturation = max(0.1, min(1.0, saturation))  # Clamp between 0.1 and 1.0
        
        # Convert to HSV color space
        # OpenCV uses H: 0-179, S: 0-255, V: 0-255
        hue = angle / 2  # Convert to OpenCV hue range
        saturation_cv = int(saturation * 255)
        value = 255  # Full brightness
        
        # Convert HSV to BGR
        hsv_color = np.array([[[hue, saturation_cv, value]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        
        # Return RGB values (convert from BGR)
        return (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
    
    return None

def draw_sliders(panel, red_value, green_value, blue_value, thickness_value, panel_width, panel_height, color_wheel_center_x, color_wheel_center_y, color_wheel_radius):
    """
    Draw thickness slider positioned under the color wheel.
    
    Creates a horizontal slider positioned below the color wheel for
    adjusting thickness values. Includes a visual circle showing the current thickness.
    
    Args:
        panel (numpy.ndarray): Panel image to draw slider on
        red_value, green_value, blue_value (int): Current RGB values (0-255) - unused
        thickness_value (int): Current thickness value (1-20)
        panel_width, panel_height (int): Panel dimensions
        color_wheel_center_x, color_wheel_center_y (int): Color wheel center coordinates
        color_wheel_radius (int): Color wheel radius for positioning
        
    Returns:
        dict: Slider coordinates for interaction detection
    """
    # Define slider dimensions and positioning
    slider_width = color_wheel_radius * 2  # Match color wheel width
    slider_height = 20
    margin = 20
    
    # Position slider below the color wheel
    slider_x = color_wheel_center_x - color_wheel_radius
    slider_y = color_wheel_center_y + color_wheel_radius + margin
    
    # Draw slider background (gray rectangle)
    cv2.rectangle(panel, (slider_x, slider_y), (slider_x + slider_width, slider_y + slider_height), (100, 100, 100), -1)
    
    # Calculate slider indicator position
    # Higher values = right position (slider goes from left to right)
    thickness_pos = int(slider_x + ((thickness_value - 1) / 19.0) * slider_width)
    
    # Draw colored indicator for the slider
    cv2.rectangle(panel, (thickness_pos - 3, slider_y - 3), (thickness_pos + 3, slider_y + slider_height + 3), (255, 255, 255), -1)
    
    # Draw visual thickness circle above the slider
    circle_center_x = slider_x + slider_width // 2
    circle_center_y = slider_y - margin - 10
    
    # Draw the thickness circle with current thickness value
    cv2.circle(panel, (circle_center_x, circle_center_y), thickness_value, (255, 255, 255), -1)
    
    # Add border to the thickness circle for visibility
    cv2.circle(panel, (circle_center_x, circle_center_y), thickness_value, (0, 0, 0), 2)
    
    # Add label for the slider
    cv2.putText(panel, "Thickness", (slider_x, slider_y + slider_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Return slider coordinates for interaction detection
    return {
        'thickness': (slider_x, slider_y, slider_x + slider_width, slider_y + slider_height)
    }

def check_slider_interaction(finger_point, slider_coords, current_values):
    """
    Check if finger is over a slider and update values based on finger position.
    
    This function detects when the finger is hovering over a slider and calculates
    the new value based on the finger's Y position within the slider area. The
    value is calculated as a percentage of the slider height, providing smooth
    and intuitive control.
    
    Args:
        finger_point (tuple): Finger coordinates (x, y) or None
        slider_coords (dict): Dictionary of slider coordinates
        current_values (tuple): Current (red, green, blue, thickness) values
        
    Returns:
        tuple: Updated (red, green, blue, thickness) values
    """
    # If no finger is detected, return current values unchanged
    if finger_point is None:
        return current_values
    
    # Unpack current values for modification
    red_value, green_value, blue_value, thickness_value = current_values
    
    # Check each slider for intersection with finger position
    for slider_name, (x1, y1, x2, y2) in slider_coords.items():
        # Check if finger is within this slider's rectangular bounds
        if x1 <= finger_point[0] <= x2 and y1 <= finger_point[1] <= y2:
            # Calculate new value based on finger's Y position within the slider
            slider_height = y2 - y1
            relative_y = (finger_point[1] - y1) / slider_height
            
            # Update the appropriate value based on which slider is being interacted with
            # For horizontal slider: Higher finger position (higher relative_x) = higher value
            if slider_name == 'thickness':
                # Calculate relative X position for horizontal slider
                relative_x = (finger_point[0] - x1) / (x2 - x1)
                # Thickness ranges from 1 to 20
                thickness_value = int(1 + (19 * relative_x))
    
    return red_value, green_value, blue_value, thickness_value
