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

def get_centroid(points):
    points = np.array(points, dtype=np.int32)
    if points.ndim == 1 and points.shape[0] == 4:
        x1, y1, x2, y2 = points
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    elif points.ndim == 2:
        centroid = tuple(np.mean(points, axis=0).astype(int))
        return (int(centroid[0]), int(centroid[1]))
    raise ValueError("Invalid shape for centroid calculation")

def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def avg_distance_to_table(human_box, table):
    human_points = [
        (human_box[0], human_box[1]),
        (human_box[2], human_box[1]),
        (human_box[0], human_box[3]),
        (human_box[2], human_box[3])
    ]
    table_points = np.array(table)
    distances = [min([cv2.norm(np.array(h) - np.array(t), cv2.NORM_L2) for t in table_points]) for h in human_points]
    return np.mean(distances)

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
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
            box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
            union_area = box1_area + box2_area - inter_area
            if union_area > 0 and (inter_area / union_area) >= threshold:
                group.append(box2)
                used.add(j)
        group = np.array(group)
        merged.append([np.min(group[:, 0]), np.min(group[:, 1]), np.max(group[:, 2]), np.max(group[:, 3])])
    return merged

# ===============================
# Load Image
image_path = "/mnt/c/Users/nongf/Desktop/CUNEX/experiment/IMG20250226133959.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")

# Load Table Labels
with open("table_labels.json", "r") as f:
    table_boxes = json.load(f)
if not table_boxes:
    print("No table labels found. Exiting.")
    exit()

# ===============================
# Load Human Detection Model
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = "model_final_2d9806.pkl"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
instances = outputs["instances"].to("cpu")
pred_classes = instances.pred_classes.numpy()
pred_boxes = instances.pred_boxes.tensor.numpy()

# ===============================
# Filter Humans Only
human_indices = np.where(pred_classes == 0)[0]
human_boxes = pred_boxes[human_indices]
merged_human_boxes = merge_overlapping_boxes(human_boxes, threshold=OVERLAP_THRESHOLD)

# ===============================
# ASSIGN HUMAN TO TABLE BASED ON POSITION AND DISTANCE
human_to_table_map = {}

for i, human_box in enumerate(merged_human_boxes):
    best_table_idx = -1
    min_distance = float('inf')
    
    for j, table in enumerate(table_boxes):
        if is_point_inside_polygon(get_centroid(human_box), table):
            best_table_idx = j
            break  # Prioritize humans inside a table area
        
        distance = avg_distance_to_table(human_box, table)
        if distance < min_distance:
            min_distance = distance
            best_table_idx = j

    human_to_table_map[i] = best_table_idx

# ===============================
# VISUALIZATION
output_image = image.copy()

# Draw tables
for i, table in enumerate(table_boxes):
    cv2.polylines(output_image, [np.array(table)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(output_image, f"Table {i}", tuple(table[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Draw humans and connections
for i, box in enumerate(merged_human_boxes):
    human_centroid = get_centroid(box)
    cv2.circle(output_image, human_centroid, 5, (255, 0, 0), -1)
    
    if human_to_table_map[i] != -1:
        table_centroid = get_centroid(table_boxes[human_to_table_map[i]])
        cv2.line(output_image, human_centroid, table_centroid, (0, 255, 0), 2)
        cv2.putText(output_image, f"Person {i} -> Table {human_to_table_map[i]}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Human-Table Mapping", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()