## Dev/Creator: tubakhxn

# Bird Detection and Tracking with YOLOv8

This project detects and tracks a flock of birds in a video using Ultralytics YOLOv8 and OpenCV. It assigns unique IDs to each bird, draws modern bounding boxes, smooths their motion, and visualizes their paths with clean overlays. The total bird count and FPS are displayed on the video.

## Features
- Detects multiple birds per frame using YOLOv8 (pretrained model)
- Tracks each bird with a unique ID and smooths bounding boxes
- Draws colored bounding boxes and short, clean motion trails
- Shows bird count and FPS on screen
- Modern, semi-transparent overlay for clarity
- Handles missing dependencies and model downloads automatically

## How to Fork and Run
1. **Fork this repository** using the GitHub interface.
2. **Clone your fork**:
   ```
   git clone https://github.com/yourusername/birds-yolov8-tracker.git
   cd birds-yolov8-tracker
   ```
3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
4. **Add your input video** as `input.mp4` in the project folder (or let the script download a sample).
5. **Run the project**:
   ```
   python main.py
   ```
6. **View the output** in `output.mp4`.

## What is this project about?
This project demonstrates real-time object detection and tracking for birds in video, using deep learning (YOLOv8) and computer vision (OpenCV). It is useful for wildlife monitoring, research, and computer vision education.

## Relevant Wikipedia Articles
- [Object detection](https://en.wikipedia.org/wiki/Object_detection)
- [Object tracking](https://en.wikipedia.org/wiki/Video_tracking)
- [YOLO (You Only Look Once)](https://en.wikipedia.org/wiki/You_Only_Look_Once_(object_detection))
- [OpenCV](https://en.wikipedia.org/wiki/OpenCV)
- [Bird migration](https://en.wikipedia.org/wiki/Bird_migration)

---

Feel free to fork, improve, and use for your own research or projects!
