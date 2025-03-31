import detectron2
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load an image
image_path = r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/IMG20250226133959.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")

# Create config for human detection
cfg = get_cfg()
cfg.merge_from_file(r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Adjust threshold if needed
cfg.MODEL.WEIGHTS = r"/mnt/c/Users/nongf/Desktop/CUNEX/experiments/model_final_2d9806.pkl"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
outputs = predictor(image)

# Get predictions
instances = outputs["instances"].to("cpu")
pred_classes = instances.pred_classes.numpy()
pred_masks = instances.pred_masks.numpy()
pred_boxes = instances.pred_boxes.tensor.numpy()

# Filter only humans (COCO class "person" = 0)
human_indices = np.where(pred_classes == 0)[0]

# Check if any humans were detected
if len(human_indices) == 0:
    print("No humans detected.")
    exit()

# Extract human bounding boxes and masks
human_boxes = pred_boxes[human_indices]
human_masks = pred_masks[human_indices]

### Table Detection (Replace with your table detection model or method)
# Assuming "table" class index in COCO is 60 (Adjust based on model)
table_indices = np.where(pred_classes == 60)[0]
if len(table_indices) == 0:
    print("No tables detected.")
    exit()

table_boxes = pred_boxes[table_indices]

### Assign Humans to the Closest Table
def get_centroid(box):
    """Calculate the centroid of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

human_centroids = np.array([get_centroid(box) for box in human_boxes])
table_centroids = np.array([get_centroid(box) for box in table_boxes])

# Assign each human to the nearest table
human_to_table_map = {}

for i, human_centroid in enumerate(human_centroids):
    distances = np.linalg.norm(table_centroids - human_centroid, axis=1)
    nearest_table_idx = np.argmin(distances)
    human_to_table_map[i] = nearest_table_idx  # Mapping human index to table index

# Visualize Results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(instances)

# Ensure output_image is a NumPy array and convert to BGR
output_image = np.array(v.get_image(), dtype=np.uint8)
output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# Resize the output image
scale_percent = 50  # Resize to 50% of original size
width = int(output_image.shape[1] * scale_percent / 100)
height = int(output_image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_output = cv2.resize(output_image, dim, interpolation=cv2.INTER_AREA)

# Scale centroids accordingly
scale_x = width / image.shape[1]
scale_y = height / image.shape[0]

scaled_human_centroids = [(int(x * scale_x), int(y * scale_y)) for x, y in human_centroids]
scaled_table_centroids = [(int(x * scale_x), int(y * scale_y)) for x, y in table_centroids]

# Draw table-human connections
for i, human_center in enumerate(scaled_human_centroids):
    table_center = scaled_table_centroids[human_to_table_map[i]]
    
    # Draw a line between human and table
    cv2.line(resized_output, human_center, table_center, (0, 255, 0), 2)
    
    # Label each human and their assigned table
    cv2.putText(resized_output, f"Person {i}", human_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(resized_output, f"Table {human_to_table_map[i]}", table_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Display the resized output image
cv2.imshow("Human-Table Mapping", resized_output)
print("Press any key to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()
