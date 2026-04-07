"""
Demo Video Generator for Pothole Detection
Creates a video showing the model detecting potholes on road images.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Paths
BASE_DIR = Path(r"C:\Users\AadithGeethMohan\OneDrive\Desktop\ACM SIGKDD TASK\pothole-detection")
MODEL_PATH = BASE_DIR / "best.pt"
DATASET_DIR = BASE_DIR / "Pothole Detection.v1i.yolov8-obb"
VAL_IMAGES = DATASET_DIR / "valid" / "images"
TEST_IMAGES = DATASET_DIR / "test" / "images"
OUTPUT_DIR = BASE_DIR / "demo_video"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading model...")
model = YOLO(str(MODEL_PATH))

# Collect images from validation + test sets
val_images = sorted(list(VAL_IMAGES.glob("*.jpg")) + list(VAL_IMAGES.glob("*.png")))
test_images = sorted(list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png")))
all_images = val_images + test_images
print(f"Found {len(val_images)} validation + {len(test_images)} test = {len(all_images)} total images")

# Select diverse set (skip some to show variety)
# Take every 3rd image to get variety, max 60 frames
selected = all_images[::3][:60]
print(f"Selected {len(selected)} images for video")

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 2  # 4 fps = each image shows for 0.25s — good for a demo
OUTPUT_VIDEO = OUTPUT_DIR / "pothole_detection_demo_tight_boxes.mp4"

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT + 60))

print(f"Generating video (conf≥0.5, tight boxes): {OUTPUT_VIDEO}")

for i, img_path in enumerate(selected):
    # Run inference
    result = model.predict(source=str(img_path), verbose=False, conf=0.5)[0]
    n_boxes = len(result.boxes) if result.boxes is not None else 0

    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

    # Draw bounding boxes
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Shrink box to 65% size (OBB->axis-aligned inflation correction)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            scale = 0.65
            x1 = int(cx - bw * scale / 2)
            y1 = int(cy - bh * scale / 2)
            x2 = int(cx + bw * scale / 2)
            y2 = int(cy + bh * scale / 2)

            # Clamp to image bounds
            x1, x2 = max(0, x1), min(FRAME_WIDTH, x2)
            y1, y2 = max(0, y1), min(FRAME_HEIGHT, y2)

            # Color based on confidence
            if conf >= 0.7:
                color = (0, 220, 60)
            elif conf >= 0.5:
                color = (0, 180, 220)
            else:
                color = (0, 120, 255)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

            # Label
            label = f"pothole {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # Add info bar at bottom
    info_bar = np.zeros((60, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(info_bar, f"YOLOv8n Pothole Detection (conf>=0.5, tight boxes)  |  Frame {i+1}/{len(selected)}  |  Detected: {n_boxes} potholes",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Combine frame + info bar
    frame = np.vstack([img, info_bar])

    writer.write(frame)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(selected)} frames...")

writer.release()
print(f"\n✅ Demo video saved: {OUTPUT_VIDEO}")
print(f"   Duration: {len(selected) / FPS:.1f} seconds at {FPS} fps")
print(f"   Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT + 60}")
