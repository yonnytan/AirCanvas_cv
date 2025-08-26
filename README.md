# Air Canvas - My First Computer Vision Project

I’ve always been interested in how computers process images and video, and I find a lot of practical use from it, whethet its Face ID on my iPhone, photo album filtering and searching, or even using AI to take in an image and generate results.

After learning the basics of computer vision in Python and going through a few tutorials, I decided to dive in and build my first computer vision project. I wanted something that would teach me a lot, but also be fun and interactive enough to keep me motivated.

## What I Built

**Air Canvas** is a hand gesture drawing application that lets you draw on a virtual canvas using just your hand and a webcam. Basically having a digital whiteboard that responds to your movements in real-time.

<img src="AirCanvasGif.gif" alt="AirCanvasDemo1" width="450"/>
<img src="AirCanvasGif2.gif" alt="AirCanvasDemo2" width="450"/>

### Key Features I Implemented:

- **Hand Tracking**: Uses MediaPipe to detect your index finger and thumb
- **Pinch-to-Draw**: Make a pinching gesture to start drawing, release to stop
- **Color Selection**: Hover over color boxes with your index finger to select different colors
- **Color Wheel**: A Color Wheel Picker for drawing with custom colors
- **Adaptive Eraser**: Automatically adjusts to work on both white and black canvases
- **Canvas Toggle**: Switch between white and black backgrounds
- **Thickness Control**: Adjust line thickness with a vertical slider

### The Journey

This project taught me about:

- **OpenCV**: How to work with video feeds, image processing, and real-time graphics
- **MediaPipe**: Using pre-trained models for hand landmark detection
- **Coordinate Systems**: The complex mapping between camera coordinates and screen coordinates
- **UI Design**: Creating intuitive interfaces that work with gesture input
- **Performance Optimization**: Making real-time applications run smoothly

## Future Improvements I'd Love to Make

- **Multiple Hand Support**: Track both hands for more complex gestures
- **Gesture Library**: Add more gestures like "undo" and "redo"
- **Brush Types**: Different brush styles (pencil, marker, spray paint, etc...)
- **Overall Performance Improvements**: More precise feedback with distracting backgrounds
- **Depth of Field**: Improve the responsiveness when one finger is farther than the other
- **Export Options**: Export drawings as PDF, PNG, JPG, or SVG

## Technical Details

- **Language**: Python 3.12
- **Main Libraries**: OpenCV, MediaPipe, NumPy
- **Architecture**: Modular design with separate modules for camera, hand tracking, canvas, and controls
- **Performance**: Real-time processing at 30+ FPS
- **Compatibility**: Works on macOS, Windows, and Linux

## How to run the project yourself!

```bash
# Install dependencies
pip install opencv-python mediapipe numpy

# Run the application
python main.py
```

I'm excited to continue learning and building more computer vision projects. This is just the beginning!
