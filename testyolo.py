import torch
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# Load image
image_path = r"all_frame/frame (1).jpg"
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

# Load YOLO model
model = YOLO("yolo11x.pt")  # Change to yolov11.pt if you have a custom model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run inference
results = model(image_path, device=device)

# Extract human detections (class 0 = person)
human_boxes = []
human_scores = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        score = float(box.conf[0])
        if cls == 0 and score > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            human_boxes.append([x1, y1, x2, y2])
            human_scores.append(score)

if not human_boxes:
    print("No humans detected.")
    exit()

# Merge overlapping boxes
def merge_overlapping_boxes(boxes, scores, iou_threshold=0.5):
    # Convert boxes to the correct format for NMSBoxes (they should be a list of boxes)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.3, nms_threshold=iou_threshold)
    
    # Convert indices to a flat list, as NMSBoxes returns a 2D array
    if len(indices) > 0:
        indices = indices.flatten()
        merged_boxes = [boxes[i] for i in indices]
        return merged_boxes
    else:
        return []


human_boxes = merge_overlapping_boxes(human_boxes, human_scores)

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

# Association functions
def compute_intersection_area(box, table):
    mask_table = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_table, [np.array(table)], 1)
    mask_box = np.zeros_like(mask_table)
    x1, y1, x2, y2 = box
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, -1)
    return np.sum(np.logical_and(mask_table, mask_box))

def get_centroid(points):
    points = np.array(points)
    if points.shape[0] == 4 and points.ndim == 1:
        x1, y1, x2, y2 = points
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    else:
        return tuple(np.mean(points, axis=0, dtype=int))

# Match humans to tables
human_to_table = {}
for i, box in enumerate(human_boxes):
    max_area = 0
    best_idx = -1
    for j, table in enumerate(table_boxes):
        area = compute_intersection_area(box, table)
        if area > max_area:
            max_area = area
            best_idx = j

    if best_idx == -1:
        h_centroid = get_centroid(box)
        best_idx = min(range(len(table_boxes)),
                       key=lambda j: np.linalg.norm(np.array(h_centroid) - np.array(get_centroid(table_boxes[j]))))

    human_to_table[i] = best_idx

# Visualization
output = image.copy()
for i, box in enumerate(human_boxes):
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    h_centroid = get_centroid(box)
    cv2.circle(output, h_centroid, 5, (255, 0, 0), -1)
    table_idx = human_to_table[i]
    t_centroid = get_centroid(table_boxes[table_idx])
    cv2.line(output, h_centroid, t_centroid, (0, 0, 255), 2)
    cv2.putText(output, f'Table {table_idx}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

for idx, table in enumerate(table_boxes):
    cv2.polylines(output, [np.array(table)], True, (255, 255, 0), 2)
    t_centroid = get_centroid(table)
    cv2.putText(output, f'Table {idx}', t_centroid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

cv2.imshow("Human-Table Association", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
