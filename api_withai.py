import detectron2
import torch
import cv2
import numpy as np
import json
import os
import csv
import time
import pandas as pd
from datetime import datetime
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# ===============================
# CONFIGURATION
RTSP_URL = "rtsp://admin:Password@1@192.168.0.65:554/Streaming/Channels/101/"  # Replace with your RTSP URL
FRAME_SKIP_INTERVAL = 1  # Process every 5 seconds
CSV_FILE = "table_occupancy_log.csv"
OVERLAP_THRESHOLD = 0.3

# Column constraints mapping
column_constraints = {
    "A": 7, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3,
    "G": 7, "H": 5, "I": 10, "J": 7, "K": 2
}

# Map table numbers to API table labels
def generate_table_mapping():
    mapping = {}
    table_index = 0
    for col, count in column_constraints.items():
        for row in range(count):
            mapping[table_index] = f"{col}{row}"
            table_index += 1
    return mapping

table_mapping = generate_table_mapping()

# ===============================
# AI MODEL LOADING
cfg = get_cfg()

cfg.merge_from_file("output/detectron2_config.yaml")
cfg.MODEL.WEIGHTS = "output/model_final.pth"

# cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "model_final_2d9806.pkl"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Load table positions
with open("table_labels_1080p.json", "r") as f:
    table_boxes = json.load(f)

# ===============================
# HELPER FUNCTIONS

def get_centroid(points):
    points = np.array(points, dtype=np.int32)
    if points.ndim == 1 and points.shape[0] == 4:
        x1, y1, x2, y2 = points
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    elif points.ndim == 2:
        return tuple(np.mean(points, axis=0).astype(int))
    raise ValueError("Invalid shape for centroid calculation")

def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def avg_distance_to_table(human_box, table):
    human_points = [(human_box[0], human_box[1]), (human_box[2], human_box[1]),
                    (human_box[0], human_box[3]), (human_box[2], human_box[3])]
    table_points = np.array(table)
    distances = [min([cv2.norm(np.array(h) - np.array(t), cv2.NORM_L2) for t in table_points]) for h in human_points]
    return np.mean(distances)
def ensure_csv_exists():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Create the header with timestamp and table labels
            header = ["timestamp"] + list(table_mapping.values())
            writer.writerow(header)

def process_frame(frame):
    ensure_csv_exists()  # Ensure CSV exists before writing data
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    pred_boxes = instances.pred_boxes.tensor.numpy()

    # Filter only humans (COCO class "person" = 0)
    # Filter Humans Only///////////////Change class number here trained model and pretrained model is not the same
    human_indices = np.where(pred_classes == 1)[0]
    human_boxes = pred_boxes[human_indices]

    # Map humans to tables
    table_counts = {table_mapping[i]: 0 for i in range(len(table_boxes))}
    human_assignments = {}

    # Draw table polygons
    # for table in table_boxes:
    #     cv2.polylines(frame, [np.array(table, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw human bounding boxes
    for human_box in human_boxes:
        x1, y1, x2, y2 = map(int, human_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Assign humans to tables
    for human_box in human_boxes:
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

        if best_table_idx != -1:
            table_label = table_mapping.get(best_table_idx, f"Unknown-{best_table_idx}")
            table_counts[table_label] += 1
            human_assignments[get_centroid(human_box)] = table_label

    # Convert counts to occupancy status
    for table, count in table_counts.items():
        if count == 0:
            table_counts[table] = 0
        elif 1 <= count <= 3:
            table_counts[table] = 1
        else:
            table_counts[table] = 2

    # Draw lines from humans to assigned tables
    for human_center, table_label in human_assignments.items():
        for j, table in enumerate(table_boxes):
            if table_mapping.get(j) == table_label:
                table_center = get_centroid(table)
                cv2.line(frame, human_center, table_center, (255, 255, 0), 2)
                color = (0, 255, 0) if table_counts[table_label] == 1 else (0, 255, 255) if table_counts[table_label] == 2 else (0, 0, 255)
                cv2.putText(frame, table_label, (human_center[0] + 10, human_center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save results to CSV
    timestamp = datetime.now().isoformat()
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp] + list(table_counts.values()))

    return frame, table_counts


# ===============================
# RUNNING AI ON RTSP
def run_ai_on_rtsp():
    # cap = cv2.VideoCapture(RTSP_URL)
    gst_pipeline = (
        f"rtspsrc location={RTSP_URL} ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
    )
    # cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture("/media/noboru/Windows/Users/nongf/Downloads/Cunex_video/2025-03-19/192.168.0.67_01_20250319121855674.mp4")
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not received")
            break

        if frame_count % (FRAME_SKIP_INTERVAL * 30) == 0:  # Approx 5-second interval (assuming 30 FPS)
            frame, table_status = process_frame(frame)
            print(f"Processed frame at {datetime.now().isoformat()}: {table_status}")

            # Show the frame with bounding boxes and table areas
            cv2.imshow("Table Occupancy Detection", frame)

            # Wait for 1 ms to update the frame and exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        time.sleep(1 / 30)  # Simulate 30 FPS

    cap.release()
    cv2.destroyAllWindows()

# ===============================
# API
app = FastAPI()
@app.get("/get_table")
async def get_latest_data():
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return JSONResponse(content={"error": "No data available"}, status_code=404)

        latest_entry = df.iloc[-1].fillna(-1)
        timestamp = latest_entry[0]  # Extract timestamp
        
        # Group columns by their prefix (A, B, C, etc.) and create lists
        table_data = {}
        for col in df.columns[1:]:  # Skip timestamp column
            table_label = col[0]  # The first letter of the column name
            if table_label not in table_data:
                table_data[table_label] = []
            table_data[table_label].append(int(latest_entry[col]))

        # Format the response as required
        response = {
            **table_data,  # Spread table data keys into the response
            "canteen_name": "icanteen",
            "timestamp": timestamp,
        }
        return JSONResponse(content=response)

    except FileNotFoundError:
        return JSONResponse(content={"error": "CSV file not found"}, status_code=404)

# ===============================
# MAIN EXECUTION
if __name__ == "__main__":
    import uvicorn
    from threading import Thread

    # Run AI in a separate thread
    ai_thread = Thread(target=run_ai_on_rtsp)
    ai_thread.start()
    uvicorn.run("api_withai:app", host="0.0.0.0", port=9926, reload=True)
