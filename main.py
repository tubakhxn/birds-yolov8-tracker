import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import CentroidTracker
from utils import get_color, smooth_bbox, draw_overlay, draw_bbox, download_sample_video

# Check and install missing dependencies
def check_dependencies():
    import importlib
    required = ['ultralytics', 'cv2', 'numpy', 'scipy', 'requests']
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg if pkg != 'cv2' else 'cv2')
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        print("Please install them with: pip install " + ' '.join(missing))
        sys.exit(1)

# Download YOLOv8 model if not present
def ensure_yolo_model(model_path='yolov8n.pt'):
    if not os.path.exists(model_path):
        print(f"Downloading YOLOv8 model to {model_path}...")
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(model_path)

# Download sample video if not present
def ensure_input_video():
    if not os.path.exists('input.mp4'):
        print("input.mp4 not found. Downloading sample bird video...")
        url = 'https://samplelib.com/mp4/sample-960x400-5mb.mp4'  # Replace with a real bird video if available
        download_sample_video(url, 'input.mp4')

# Main processing function
def main():
    check_dependencies()
    ensure_yolo_model()
    ensure_input_video()

    model = YOLO('yolov8n.pt')
    tracker = CentroidTracker(maxDisappeared=20, maxTrail=10)  # Shorter trails for clarity
    cap = cv2.VideoCapture('input.mp4')
    if not cap.isOpened():
        print("Error: Could not open input.mp4")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    prev_bboxes = {}
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Run YOLOv8 detection
        results = model(frame, verbose=False)[0]
        # Filter for 'bird' class (COCO class 14)
        bird_indices = [i for i, c in enumerate(results.boxes.cls.cpu().numpy()) if int(c) == 14]
        bboxes = []
        for i in bird_indices:
            box = results.boxes.xyxy[i].cpu().numpy().astype(int)
            bboxes.append(box.tolist())
        # Track birds
        objects, trails = tracker.update(bboxes)
        # Draw overlay
        vis = draw_overlay(frame, alpha=0.15)
        # Draw birds
        for objectID, (centroid, bbox) in objects.items():
            color = get_color(objectID)
            # Smooth bbox
            prev = prev_bboxes.get(objectID)
            smoothed = smooth_bbox(prev, bbox, alpha=0.7)
            prev_bboxes[objectID] = smoothed
            # Draw bbox
            draw_bbox(vis, smoothed, color, label=f"Bird {objectID}")
            # Draw clean, short trail
            pts = list(trails[objectID])
            if len(pts) > 2:
                for j in range(1, len(pts)):
                    if pts[j-1] is None or pts[j] is None:
                        continue
                    fade = int(120 * (1 - j / len(pts)))
                    trail_color = tuple(int(0.7*c + 0.3*255) for c in color)
                    cv2.line(vis, tuple(pts[j-1]), tuple(pts[j]), trail_color, 2)
            # Draw a fading circle at the last position
            if len(pts) > 0:
                last = pts[-1]
                cv2.circle(vis, tuple(last), 7, color, -1)
                cv2.circle(vis, tuple(last), 12, color, 2)
        # Show bird count
        cv2.putText(vis, f"Birds: {len(objects)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        # Show FPS
        elapsed = time.time() - start_time
        fps_disp = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"FPS: {fps_disp:.1f}", (width-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        out.write(vis)
        # Optional: show live preview
        # cv2.imshow('Bird Tracking', vis)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print("Processing complete. Output saved as output.mp4")

if __name__ == "__main__":
    main()
