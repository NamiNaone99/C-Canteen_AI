import torch
import cv2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
import os

# Load the trained model
cfg = get_cfg()
cfg.merge_from_file("output/detectron2_config.yaml")
cfg.MODEL.WEIGHTS = "output/detectron2_model.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for predictions
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Image path variable
imgpath = "all_frame/frame (1).jpg"  # Change this to your image path

# Function to perform inference
def run_inference(image_path):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    # Visualize results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]
    
    # Save and show the result
    output_path = "output/inference_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Inference result saved to {output_path}")
    
    cv2.imshow("Inference Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run inference
if __name__ == "__main__":
    run_inference(imgpath)