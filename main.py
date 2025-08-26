"""
Air Canvas - Hand Gesture Drawing Application

This application allows users to draw on a virtual canvas using hand gestures detected
through a webcam. The interface is split into two panels: a drawing canvas on the left
and a camera feed with controls on the right.

Key Features:
- Hand tracking using MediaPipe for finger and thumb detection
- Pinch gesture recognition for drawing activation
- Color selection through hover interaction
- RGB sliders for custom color creation
- Canvas background toggle (white/black)
- Adaptive eraser that changes color based on canvas background

Design Decisions:
- Portrait crop of camera feed provides better hand visibility
- Split-screen layout separates drawing from controls
- Mouse click for canvas toggle provides alternative to gesture control
- Coordinate mapping handles camera-to-panel transformation
"""

import cv2
from camera import open_camera, get_frame
from handtracker import get_fingertip, is_pinch, map_to_panel_coordinates
from canvas import create_canvas, draw_line, overlay_canvas
from controls import check_color_selection, draw_sliders, check_slider_interaction, draw_color_wheel, check_color_wheel_interaction
import numpy as np

def get_screen_size():
    """
    Get the screen size to properly size the application window.
    
    Uses system_profiler on macOS to detect actual screen resolution.
    Falls back to common resolution if detection fails.
    
    Returns:
        tuple: (width, height) in pixels
    """
    import subprocess
    import re
    
    try:
        # macOS-specific: system_profiler provides detailed display information
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True)
        
        # Parse the output to find resolution information
        output = result.stdout
        resolution_match = re.search(r'Resolution:\s*(\d+)\s*x\s*(\d+)', output)
        
        if resolution_match:
            width = int(resolution_match.group(1))
            height = int(resolution_match.group(2))
            return width, height
    except:
        pass
    
    # Fallback: Use common resolution that works on most modern displays
    return 1920, 1080

def crop_camera_to_portrait(frame):
    """
    Crop the camera frame to portrait orientation for better hand visibility.
    
    Most webcams provide landscape video, but portrait orientation works better
    for hand tracking as it matches natural hand positioning. This function:
    1. Crops a square from the center of the frame
    2. Further crops to 3:4 aspect ratio (portrait)
    
    Args:
        frame: Input camera frame (landscape orientation)
        
    Returns:
        Cropped frame in portrait orientation
    """
    height, width = frame.shape[:2]
    
    # Use the smaller dimension to ensure we can crop a square
    crop_size = min(width, height)
    
    # Center the square crop to capture the most important area
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    
    # Extract the square portion
    square_crop = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    # Convert square to portrait (3:4 aspect ratio) for better hand tracking
    portrait_height = crop_size
    portrait_width = int(crop_size * 3 / 4)  # 3:4 ratio
    
    # Center the portrait crop within the square
    start_x_portrait = (crop_size - portrait_width) // 2
    start_y_portrait = 0
    
    # Final portrait crop
    portrait_crop = square_crop[start_y_portrait:start_y_portrait + portrait_height, 
                               start_x_portrait:start_x_portrait + portrait_width]
    
    return portrait_crop

# ===== APPLICATION SETUP =====
print("Starting Air Canvas...")

# Initialize camera with error handling
print("Opening camera...")
camera = open_camera()
first_frame = get_frame(camera)
first_frame = cv2.flip(first_frame, 1)  # Mirror for natural interaction

# ===== APPLICATION STATE =====
# Drawing state variables - these persist across frames
last_drawing_point = None  # Previous drawing position for line continuity
is_erasing = False  # Whether eraser mode is active
current_drawing_color = (0, 0, 0)  # Current drawing color (BGR format)
canvas_is_white = True  # Canvas background color state
selected_color_name = None  # Name of the currently selected color button

# Calculate window dimensions based on screen size
screen_width, screen_height = get_screen_size()
print(f"Screen size: {screen_width}x{screen_height}")

# Create a more reasonable window size instead of using full screen width
# Use a fixed width that works well for most screens
target_window_width = 1600  # Reasonable width for the application
target_window_height = 1000  # Reasonable height for the application

def calculate_panel_layout(total_width, total_height, left_flex=1, right_flex=1):
    """
    Calculate panel dimensions like flexbox
    
    Args:
        total_width (int): Total window width
        total_height (int): Total window height
        left_flex (int): Flex value for left panel (default: 1)
        right_flex (int): Flex value for right panel (default: 1)
        
    Returns:
        dict: Panel dimensions for left and right panels
    """
    total_flex = left_flex + right_flex
    
    left_width = int((left_flex / total_flex) * total_width)
    right_width = total_width - left_width
    
    return {
        'left': {'width': left_width, 'height': total_height},
        'right': {'width': right_width, 'height': total_height}
    }

# Calculate panel layout using flexbox-like system
# You can easily adjust the flex values to change panel ratios
# Examples: (1,1) = 50/50 split, (2,1) = 66/33 split, (1,2) = 33/66 split
layout = calculate_panel_layout(target_window_width, target_window_height, 1, 1)

left_panel_width = layout['left']['width']
left_panel_height = layout['left']['height']

# Drawing area matches left panel dimensions
drawing_area_height = left_panel_height
drawing_area_width = left_panel_width

# Initialize color control values
red_value = 0
green_value = 0
blue_value = 0
line_thickness = 5

# Color wheel parameters
color_wheel_center_x = 600  # Will be positioned on the right panel
color_wheel_center_y = 150
color_wheel_radius = 80

# Define color button layout and properties
# Each button is defined as (coordinates, default_color)
# Coordinates: (x1, y1, x2, y2) where (x1,y1) is top-left, (x2,y2) is bottom-right
color_buttons = {
    "red": ((10, 10, 80, 80), (0, 0, 255)),
    "green": ((90, 10, 160, 80), (0, 255, 0)),
    "blue": ((170, 10, 240, 80), (255, 0, 0)),
    "yellow": ((250, 10, 320, 80), (0, 255, 255)),
    "eraser": ((330, 10, 400, 80), (255, 255, 255)),  # Color will be dynamic
    "custom": ((410, 10, 480, 80), (0, 0, 0))  # Shows current RGB values
}

# Initialize drawing canvas with white background
print("Creating drawing canvas...")
drawing_canvas = create_canvas(drawing_area_height, drawing_area_width, (255, 255, 255))

# Get camera properties for coordinate mapping
print("Setting up camera view...")
test_frame = get_frame(camera)
test_frame = cv2.flip(test_frame, 1)
cropped_camera_view = crop_camera_to_portrait(test_frame)
camera_height, camera_width = cropped_camera_view.shape[:2]

# Right panel dimensions from flexbox layout
right_panel_width = layout['right']['width']
right_panel_height = layout['right']['height']

print(f"Window size: {target_window_width}x{target_window_height}")
print(f"Left panel: {left_panel_width}x{left_panel_height}")
print(f"Right panel: {right_panel_width}x{right_panel_height}")
print(f"Panel ratio: {layout['left']['width']}:{layout['right']['width']}")

print(f"Camera size: {camera_width}x{camera_height}")
print(f"Drawing area: {drawing_area_width}x{drawing_area_height}")

# ===== MOUSE INTERACTION HANDLING =====
def handle_mouse_click(event, x, y, flags, param):
    """
    Handle mouse clicks for canvas background toggle.
    
    Provides an alternative to gesture-based control for accessibility.
    Only responds to left clicks on the toggle button area.
    
    Args:
        event: OpenCV mouse event type
        x, y: Mouse coordinates in window space
        flags: Additional event flags
        param: Additional parameters (unused)
    """
    global canvas_is_white, drawing_canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only process clicks on the left panel (drawing area)
        if x < left_panel_width:
            # Define toggle button area
            button_x, button_y = 10, 10
            button_width, button_height = 120, 40
            
            # Check if click is within toggle button bounds
            if (button_x <= x <= button_x + button_width and 
                button_y <= y <= button_y + button_height):
                # Toggle canvas background and recreate canvas
                canvas_is_white = not canvas_is_white
                drawing_canvas = create_canvas(drawing_area_height, drawing_area_width, 
                                            (255, 255, 255) if canvas_is_white else (0, 0, 0))
                print(f"Canvas switched to {'white' if canvas_is_white else 'black'}")

# Initialize window and mouse callback
cv2.namedWindow("Air Canvas Split View", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Air Canvas Split View", handle_mouse_click)

# ===== MAIN APPLICATION LOOP =====
print("Starting main loop...")
while True:
    # Get current camera frame and mirror for natural interaction
    camera_frame = get_frame(camera)
    camera_frame = cv2.flip(camera_frame, 1)
    
    # Crop to portrait orientation for better hand tracking
    cropped_camera_view = crop_camera_to_portrait(camera_frame)
    
    # ===== HAND TRACKING AND GESTURE DETECTION =====
    # Detect index finger and thumb positions
    index_finger_point, thumb_point = get_fingertip(camera_frame)
    
    # Check for pinch gesture (drawing activation)
    is_pinching = is_pinch(index_finger_point, thumb_point, camera_frame.shape[1]) if index_finger_point and thumb_point else False
    
    # ===== COORDINATE MAPPING =====
    # Convert camera coordinates to drawing coordinates
    # This is necessary because the camera view is cropped and the drawing area
    # has different dimensions than the camera frame
    if index_finger_point:
        # Get camera dimensions for coordinate calculations
        camera_height, camera_width = camera_frame.shape[:2]
        
        # Calculate the crop boundaries
        crop_size = min(camera_width, camera_height)
        crop_start_x = (camera_width - crop_size) // 2
        crop_start_y = (camera_height - crop_size) // 2
        
        # Calculate portrait crop boundaries
        portrait_width = int(crop_size * 3 / 4)
        portrait_start_x = (crop_size - portrait_width) // 2
        
        # Check if finger is within the crop area
        if (crop_start_x <= index_finger_point[0] <= crop_start_x + crop_size and 
            crop_start_y <= index_finger_point[1] <= crop_start_y + crop_size):
            
            # Convert to crop-relative coordinates
            crop_x = index_finger_point[0] - crop_start_x
            crop_y = index_finger_point[1] - crop_start_y
            
            # Check if finger is within the portrait crop area
            if (portrait_start_x <= crop_x <= portrait_start_x + portrait_width):
                # Convert to portrait coordinates
                portrait_x = crop_x - portrait_start_x
                portrait_y = crop_y
                
                # Visual feedback: show finger position on camera view
                cv2.circle(cropped_camera_view, (portrait_x, portrait_y), 10, (0, 255, 0), -1)
                
                # Scale coordinates to drawing area dimensions
                drawing_x = int(portrait_x * drawing_area_width / cropped_camera_view.shape[1])
                drawing_y = int(portrait_y * drawing_area_height / cropped_camera_view.shape[0])
                drawing_point = (drawing_x, drawing_y)
            else:
                drawing_point = None
        else:
            drawing_point = None
    else:
        drawing_point = None
    
    # ===== THUMB TRACKING AND PINCH VISUALIZATION =====
    # Show thumb position and pinch line for debugging
    if thumb_point:
        # Apply same coordinate mapping as index finger
        camera_height, camera_width = camera_frame.shape[:2]
        crop_size = min(camera_width, camera_height)
        crop_start_x = (camera_width - crop_size) // 2
        crop_start_y = (camera_height - crop_size) // 2
        portrait_width = int(crop_size * 3 / 4)
        portrait_start_x = (crop_size - portrait_width) // 2
        
        if (crop_start_x <= thumb_point[0] <= crop_start_x + crop_size and 
            crop_start_y <= thumb_point[1] <= crop_start_y + crop_size):
            crop_x = thumb_point[0] - crop_start_x
            crop_y = thumb_point[1] - crop_start_y
            
            if (portrait_start_x <= crop_x <= portrait_start_x + portrait_width):
                portrait_x = crop_x - portrait_start_x
                portrait_y = crop_y
                # Visual feedback: blue circle for thumb
                cv2.circle(cropped_camera_view, (portrait_x, portrait_y), 10, (255, 0, 0), -1)
                
                # Draw pinch line when both finger and thumb are detected
                if drawing_point:
                    drawing_x, drawing_y = drawing_point
                    # Convert drawing coordinates back to camera space for line drawing
                    portrait_x_idx = int(drawing_x * cropped_camera_view.shape[1] / drawing_area_width)
                    portrait_y_idx = int(drawing_y * cropped_camera_view.shape[0] / drawing_area_height)
                    
                    # Red line indicates active pinch gesture
                    if is_pinching:
                        cv2.line(cropped_camera_view, (portrait_x, portrait_y), 
                                (portrait_x_idx, portrait_y_idx), (0, 0, 255), 3)
    
    # ===== LEFT PANEL - DRAWING AREA =====
    # Create copy of canvas for display (avoids modifying original)
    left_panel = drawing_canvas.copy()
    
    # Draw canvas background toggle button
    button_width = 120
    button_height = 40
    button_x = 10
    button_y = 10
    
    # Button styling
    cv2.rectangle(left_panel, (button_x, button_y), 
                  (button_x + button_width, button_y + button_height), (100, 100, 100), -1)
    
    # Dynamic button text based on current canvas state
    button_text = "White" if canvas_is_white else "Black"
    cv2.putText(left_panel, button_text, (button_x + 10, button_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Button border for visual clarity
    cv2.rectangle(left_panel, (button_x, button_y), 
                  (button_x + button_width, button_y + button_height), (0, 0, 0), 2)
    
    # ===== DRAWING LOGIC =====
    # Only draw when pinching and finger is in drawing area
    if is_pinching and drawing_point:
        if last_drawing_point is not None:
            if is_erasing:
                # Eraser color is opposite of canvas background
                eraser_color = (255, 255, 255) if canvas_is_white else (0, 0, 0)
                draw_line(drawing_canvas, last_drawing_point, drawing_point, eraser_color, line_thickness)
            else:
                # Draw with current color
                draw_line(drawing_canvas, last_drawing_point, drawing_point, current_drawing_color, line_thickness)
        last_drawing_point = drawing_point
    else:
        # Reset drawing point when not pinching
        last_drawing_point = None
    
    # ===== RIGHT PANEL - CAMERA VIEW WITH OVERLAY =====
    # Overlay drawing on camera view for visual feedback
    background_color = (255, 255, 255) if canvas_is_white else (0, 0, 0)
    camera_with_drawing = overlay_canvas(cropped_camera_view, drawing_canvas, background_color)
    right_panel = cv2.resize(camera_with_drawing, (right_panel_width, right_panel_height))
    
    # ===== COLOR BUTTON RENDERING =====
    # Calculate eraser color based on canvas background
    # Eraser should be the same color as canvas background for effective erasing
    eraser_color = (255, 255, 255) if canvas_is_white else (0, 0, 0)
    
    # Update the eraser button color in the color_buttons dictionary
    # This ensures the eraser button always shows the correct color
    color_buttons["eraser"] = (color_buttons["eraser"][0], eraser_color)
    
    # Draw all color buttons with appropriate colors
    for button_name, (button_coords, button_color) in color_buttons.items():
        x1, y1, x2, y2 = button_coords
        if button_name == "custom":
            # Custom button shows current RGB values
            cv2.rectangle(right_panel, (x1, y1), (x2, y2), (blue_value, green_value, red_value), -1)
        elif button_name == "eraser":
            # Eraser button background (same as canvas background)
            cv2.rectangle(right_panel, (x1, y1), (x2, y2), eraser_color, -1)
            
            # Draw eraser icon (eraser symbol) instead of just a color box
            # Calculate center of the eraser button
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw a simple eraser icon: rectangle with diagonal lines
            # Main eraser body (slightly smaller than button)
            icon_margin = 8
            icon_x1 = x1 + icon_margin
            icon_y1 = y1 + icon_margin
            icon_x2 = x2 - icon_margin
            icon_y2 = y2 - icon_margin
            
            # Draw eraser body with opposite color of canvas for visibility
            icon_color = (0, 0, 0) if canvas_is_white else (255, 255, 255)
            cv2.rectangle(right_panel, (icon_x1, icon_y1), (icon_x2, icon_y2), icon_color, 2)
            
            # Draw diagonal lines to represent eraser strokes
            line_color = (100, 100, 100)  # Gray lines for visibility
            cv2.line(right_panel, (icon_x1 + 5, icon_y1 + 5), (icon_x2 - 5, icon_y2 - 5), line_color, 1)
            cv2.line(right_panel, (icon_x1 + 10, icon_y1 + 5), (icon_x2 - 5, icon_y2 - 10), line_color, 1)
            cv2.line(right_panel, (icon_x1 + 5, icon_y1 + 10), (icon_x2 - 10, icon_y2 - 5), line_color, 1)
        else:
            # Standard color buttons
            cv2.rectangle(right_panel, (x1, y1), (x2, y2), button_color, -1)
    
    # ===== COLOR WHEEL RENDERING =====
    # Draw color wheel for custom color selection
    # Position color wheel on the right side of the panel
    wheel_x = right_panel_width - color_wheel_radius - 50
    wheel_y = color_wheel_center_y
    current_rgb_color = (red_value, green_value, blue_value)
    color_wheel_coords = draw_color_wheel(right_panel, wheel_x, wheel_y, color_wheel_radius, 
                                         (blue_value, green_value, red_value))  # BGR format for OpenCV
    
    # ===== SLIDER RENDERING =====
    # Draw thickness slider positioned under the color wheel
    slider_coords = draw_sliders(right_panel, red_value, green_value, blue_value, line_thickness, 
                                right_panel_width, right_panel_height, wheel_x, wheel_y, color_wheel_radius)
    
    # ===== INTERACTION DETECTION =====
    # Convert finger position to panel coordinates for interaction detection
    panel_finger_point = map_to_panel_coordinates(index_finger_point, camera_frame, cropped_camera_view, 
                                                 right_panel_width, right_panel_height)
    
    # Check for color button hover and selection
    hovered_color, is_erasing, selected_color = check_color_selection(panel_finger_point, color_buttons)
    
    # Check for color wheel interaction and update RGB values
    wheel_color = check_color_wheel_interaction(panel_finger_point, color_wheel_coords)
    if wheel_color:
        red_value, green_value, blue_value = wheel_color
    
    # Check for slider interaction and update thickness value
    _, _, _, line_thickness = check_slider_interaction(panel_finger_point, 
                                                      slider_coords, 
                                                      (red_value, green_value, blue_value, line_thickness))
    
    # ===== COLOR STATE MANAGEMENT =====
    # Update drawing color based on user selection
    if selected_color:
        # Update the selected color name when hovering over a button
        selected_color_name = hovered_color
        
        if hovered_color == "eraser":
            is_erasing = True
        elif hovered_color == "custom":
            is_erasing = False
            # Use RGB slider values for custom color (BGR format for OpenCV)
            current_drawing_color = (blue_value, green_value, red_value)
        else:
            is_erasing = False
            # Use predefined button color
            current_drawing_color = selected_color
    
    # Apply the selected color even when not hovering (persistent selection)
    if selected_color_name and not is_erasing:
        if selected_color_name == "custom":
            # Use RGB slider values for custom color (BGR format for OpenCV)
            current_drawing_color = (blue_value, green_value, red_value)
        elif selected_color_name in color_buttons:
            # Use the predefined button color
            current_drawing_color = color_buttons[selected_color_name][1]
    
    # ===== VISUAL FEEDBACK =====
    # Highlight selected color button with borders
    # Show hover state (white border) and selected state (thicker border)
    if hovered_color:
        x1, y1, x2, y2 = color_buttons[hovered_color][0]
        # White outer border for hover visibility
        cv2.rectangle(right_panel, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 4)
        # Black inner border for contrast
        cv2.rectangle(right_panel, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # Show selected color button with a different border style
    if selected_color_name and selected_color_name != hovered_color:
        x1, y1, x2, y2 = color_buttons[selected_color_name][0]
        # Green border for selected state (when not hovering)
        cv2.rectangle(right_panel, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 3)
    
    # ===== STATUS DISPLAY =====
    # Show current RGB and thickness values
    cv2.putText(right_panel, f"RGB({red_value},{green_value},{blue_value}) Thick:{line_thickness}", 
                (10, right_panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # Show pinch status and drawing coordinates for debugging
    pinch_text = f"Pinch: {is_pinching}"
    if drawing_point:
        pinch_text += f" Drawing: {drawing_point}"
    cv2.putText(right_panel, pinch_text, (10, right_panel_height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Show current color selection status
    if selected_color_name == "eraser":
        color_text = "Selected: Eraser"
    elif selected_color_name == "custom":
        color_text = f"Selected: Custom RGB({red_value},{green_value},{blue_value})"
    elif selected_color_name:
        color_text = f"Selected: {selected_color_name.capitalize()}"
    else:
        color_text = f"Current: RGB({red_value},{green_value},{blue_value})"
    cv2.putText(right_panel, color_text, (10, right_panel_height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    # ===== USER INSTRUCTIONS =====
    
    # Labels for buttons
    cv2.putText(right_panel, "Custom", (410, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(right_panel, "Eraser", (330, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    # Label for color wheel
    wheel_label_x = wheel_x - 30
    wheel_label_y = wheel_y - color_wheel_radius - 10
    cv2.putText(right_panel, "Color Wheel", (wheel_label_x, wheel_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    # ===== DISPLAY COMPOSITION =====
    # Combine left and right panels side by side
    left_panel_resized = cv2.resize(left_panel, (left_panel_width, left_panel_height))
    combined_view = np.hstack((left_panel_resized, right_panel))
    
    # ===== WINDOW DISPLAY =====
    # Calculate the actual content width to avoid extra space
    actual_content_width = left_panel_width + right_panel_width
    
    # Show the combined interface
    cv2.imshow("Air Canvas Split View", combined_view)
    cv2.resizeWindow("Air Canvas Split View", actual_content_width, left_panel_height)
    
    # ===== KEYBOARD CONTROLS =====
    # Handle keyboard input for additional controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Clear canvas with current background color
        drawing_canvas[:] = (255, 255, 255) if canvas_is_white else (0, 0, 0)
        print("Canvas cleared")
    elif key == ord('t'):
        # Toggle canvas background color
        canvas_is_white = not canvas_is_white
        drawing_canvas = create_canvas(drawing_area_height, drawing_area_width, 
                                    (255, 255, 255) if canvas_is_white else (0, 0, 0))
        print(f"Canvas toggled to {'white' if canvas_is_white else 'black'}")
    elif key == 27:  # ESC key
        print("Exiting...")
        break

# ===== APPLICATION CLEANUP =====
print("Closing camera and windows...")
camera.release()
cv2.destroyAllWindows()
print("Air Canvas closed!")
