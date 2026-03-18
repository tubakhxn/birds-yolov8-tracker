import os
import random
import numpy as np
import cv2

# Assign a unique color for each ID
def get_color(idx):
    random.seed(idx)
    return tuple(int(x) for x in np.random.choice(range(50, 230), size=3))

# Smooth bounding boxes using exponential moving average
def smooth_bbox(prev_bbox, curr_bbox, alpha=0.6):
    if prev_bbox is None:
        return curr_bbox
    return [int(alpha * p + (1 - alpha) * c) for p, c in zip(prev_bbox, curr_bbox)]

# Draw a semi-transparent overlay
def draw_overlay(frame, alpha=0.2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# Draw modern bounding box with label
def draw_bbox(frame, bbox, color, label=None, thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    if label:
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

# Download a sample bird video if not present
def download_sample_video(url, out_path):
    import requests
    response = requests.get(url, stream=True)
    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return out_path
