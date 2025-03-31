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

# Create config
cfg = get_cfg()
cfg.merge_from_file(r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # Adjust threshold if needed
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1  # Adjust for more detections

# Load model weights from local path (after manual download)
cfg.MODEL.WEIGHTS = r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/model_final_2d9806.pkl"

# Ensure CUDA is available, otherwise use CPU
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

# Filter only human (COCO class "person" = 0)
human_indices = np.where(pred_classes == 0)[0]  # Faster than list comprehension

# Check if any humans were detected
if len(human_indices) == 0:
    print("No humans detected.")
    exit()

# Extract only human data
human_boxes = pred_boxes[human_indices]
human_masks = pred_masks[human_indices]

# Draw segmentation masks & bounding boxes
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(instances[human_indices])

# Convert the image back to BGR for OpenCV
output_image = v.get_image()[:, :, ::-1]
# Resize the output image to 80% of its original size
scale_percent = 50  # Resize to 80% of original size
width = int(output_image.shape[1] * scale_percent / 100)
height = int(output_image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized_output = cv2.resize(output_image, dim, interpolation=cv2.INTER_AREA)

# Display the resized output image
cv2.imshow("Human Segmentation", resized_output)
print("Press any key to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()