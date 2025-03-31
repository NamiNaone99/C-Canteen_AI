import detectron2
import torch
import cv2
import numpy as np
import json
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ===============================
# CONFIG: Overlap Threshold (0.3 = 30%)
OVERLAP_THRESHOLD = 0.3
# ===============================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

def merge_overlapping_boxes(boxes, threshold=0.3):
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue
        box1 = boxes[i]
        group = [box1]
        used.add(i)
        for j in range(i+1, len(boxes)):
            if j in used:
                continue
            box2 = boxes[j]
            iou = compute_iou(box1, box2)
            if iou >= threshold:
                group.append(box2)
                used.add(j)
        # Merge group into one box (min-max)
        group = np.array(group)
        x1 = np.min(group[:, 0])
        y1 = np.min(group[:, 1])
        x2 = np.max(group[:, 2])
        y2 = np.max(group[:, 3])
        merged.append([x1, y1, x2, y2])
    return merged

# ===============================
# Load Image
image_path = r"all_frame/frame (1).jpg"
image = cv2.imread(image_path)
original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")
image = original_image.copy()
if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")

# Load or create table label file
table_label_path = "table_labels_1080p.json"
table_boxes = []
if os.path.exists(table_label_path):
    use_saved = input("Use saved table labels? (y/n): ")
    if use_saved.lower() == 'y':
        with open(table_label_path, "r") as f:
            table_boxes = json.load(f)

# ===============================
# Load Human Detection Model
cfg = get_cfg()
cfg.merge_from_file(r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/model_final_2d9806.pkl"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
outputs = predictor(image)

instances = outputs["instances"].to("cpu")
pred_classes = instances.pred_classes.numpy()
pred_masks = instances.pred_masks.numpy()
pred_boxes = instances.pred_boxes.tensor.numpy()

# ===============================
# Filter Only Humans
human_indices = np.where(pred_classes == 0)[0]
if len(human_indices) == 0:
    print("No humans detected.")
    exit()

human_boxes = pred_boxes[human_indices]
human_masks = pred_masks[human_indices]

# ===============================
# MERGE OVERLAPPING HUMANS
merged_human_boxes = merge_overlapping_boxes(human_boxes, threshold=OVERLAP_THRESHOLD)

# ===============================
# MANUAL TABLE SELECTION
if not table_boxes:
    selected_corners = []
    
    def click_event(event, x, y, flags, param):
        global selected_corners, table_boxes, image
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_corners.append((x, y))
            print(f"Point selected: {x}, {y}")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Tables", image)

        elif event == cv2.EVENT_RBUTTONDOWN and selected_corners:
            selected_corners.pop()
            print("Undo last point")
            image = original_image.copy()
            for table in table_boxes:
                for point in table:
                    cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)  # Keep saved points visible
            for point in selected_corners:
                cv2.circle(image, point, 5, (0, 255, 0), -1)
            cv2.imshow("Select Tables", image)
    
    print("Click to define table corners, press ENTER to save, BACKSPACE to undo, ESC to finish.")
    cv2.imshow("Select Tables", image)
    cv2.setMouseCallback("Select Tables", click_event)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(selected_corners) > 2:
            table_boxes.append(selected_corners[:])
            for point in selected_corners:
                cv2.circle(image, point, 5, (0, 0, 255), -1)  # Change dot color when saved
            selected_corners = []
            print("Table saved.")
        elif key == 8 and selected_corners:
            selected_corners.pop()
            print("Undo last point.")
            image = original_image.copy()
            for table in table_boxes:
                for point in table:
                    cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)  # Keep saved points visible
            for point in selected_corners:
                cv2.circle(image, point, 5, (0, 255, 0), -1)
            cv2.imshow("Select Tables", image)
        elif key == 27:
            break
    
    cv2.destroyWindow("Select Tables")
    with open(table_label_path, "w") as f:
        json.dump(table_boxes, f)
        print("Table labels saved.")

if len(table_boxes) == 0:
    print("No tables selected.")
    exit()

# ===============================
# HELPER FUNCTIONS
def compute_intersection_area(mask, table):
    table_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(table_mask, [np.array(table)], 1)
    intersection = np.logical_and(mask, table_mask)
    return np.sum(intersection)

def get_centroid(points):
    points = np.array(points)
    if points.ndim == 1 and points.shape[0] == 4:
        x1, y1, x2, y2 = points
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    else:
        return tuple(np.mean(points, axis=0).astype(int))

# ===============================
# ASSIGN HUMAN TO TABLE
human_to_table_map = {}
for i, human_box in enumerate(merged_human_boxes):
    # No longer using mask area, just centroid distance (you can keep mask logic if needed)
    human_centroid = get_centroid(human_box)
    min_distance = float('inf')
    best_table_idx = -1
    for j, table in enumerate(table_boxes):
        table_centroid = get_centroid(table)
        distance = np.linalg.norm(np.array(human_centroid) - np.array(table_centroid))
        if distance < min_distance:
            min_distance = distance
            best_table_idx = j
    human_to_table_map[i] = best_table_idx

# ===============================
# VISUALIZATION
output_image = image.copy()
for i, table in enumerate(table_boxes):
    cv2.polylines(output_image, [np.array(table)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(output_image, f"Table {i}", tuple(table[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

for i, box in enumerate(merged_human_boxes):
    human_centroid = get_centroid(box)
    table_centroid = get_centroid(table_boxes[human_to_table_map[i]])
    cv2.rectangle(output_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.circle(output_image, tuple(map(int, human_centroid)), 5, (255, 0, 0), -1)
    cv2.circle(output_image, tuple(map(int, table_centroid)), 5, (0, 255, 0), -1)

    cv2.line(output_image, tuple(map(int, human_centroid)), tuple(map(int, table_centroid)), (0, 255, 0), 2)
    cv2.putText(output_image, f"Person {i} -> Table {human_to_table_map[i]}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Human-Table Mapping", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
