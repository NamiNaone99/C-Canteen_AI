import torch
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# Load image/experiment
image_path = r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/IMG20250226133959.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Load or create table labels
table_label_path = "table_labels.json"
table_boxes = []
if os.path.exists(table_label_path):
    use_saved = input("Use saved table labels? (y/n): ")
    if use_saved.lower() == 'y':
        with open(table_label_path, "r") as f:
            table_boxes = json.load(f)

# Load YOLOv12 segmentation model
model = YOLO("yolo11x-seg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run inference
results = model(image_path, device=device)

# Extract human segmentations (class 0 = person)
human_masks = []
human_scores = []

for r in results:
    for mask, box in zip(r.masks.xy, r.boxes):
        cls = int(box.cls[0])
        score = float(box.conf[0])
        if cls == 0 and score > 0.3:
            human_masks.append(mask)
            human_scores.append(score)

if not human_masks:
    print("No humans detected.")
    exit()

# Manual Table Selection
if not table_boxes:
    selected_corners = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_corners.append((x, y))
            print(f"Selected point: {x}, {y}")
        elif event == cv2.EVENT_RBUTTONDOWN and selected_corners:
            selected_corners.pop()
            print("Removed last point")

    print("Click to define table corners (ENTER to save, BACKSPACE to undo, ESC to finish)")
    cv2.imshow("Select Tables", image)
    cv2.setMouseCallback("Select Tables", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(selected_corners) > 2:
            table_boxes.append(selected_corners.copy())
            selected_corners = []
            print("Table saved.")
        elif key == 8 and selected_corners:
            selected_corners.pop()
        elif key == 27:
            break

    cv2.destroyWindow("Select Tables")
    with open(table_label_path, "w") as f:
        json.dump(table_boxes, f)

if not table_boxes:
    print("No tables defined.")
    exit()

# Compute intersection area with masks
def compute_intersection_area(mask, table):
    mask_table = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask_table, [np.array(table)], 1)
    mask_human = np.zeros_like(mask_table)
    cv2.fillPoly(mask_human, [np.array(mask, dtype=np.int32)], 1)
    return np.sum(np.logical_and(mask_table, mask_human))

# Match humans to tables
human_to_table = {}
for i, mask in enumerate(human_masks):
    max_area = 0
    best_idx = -1
    for j, table in enumerate(table_boxes):
        area = compute_intersection_area(mask, table)
        if area > max_area:
            max_area = area
            best_idx = j

    human_to_table[i] = best_idx

# Visualization
output = image.copy()
for i, mask in enumerate(human_masks):
    cv2.polylines(output, [np.array(mask, dtype=np.int32)], True, (0, 255, 0), 2)
    table_idx = human_to_table[i]
    cv2.putText(output, f'Table {table_idx}', tuple(np.array(mask).mean(axis=0, dtype=int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

for idx, table in enumerate(table_boxes):
    cv2.polylines(output, [np.array(table)], True, (255, 255, 0), 2)
    cv2.putText(output, f'Table {idx}', tuple(np.array(table).mean(axis=0, dtype=int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imshow("Human-Table Association", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
