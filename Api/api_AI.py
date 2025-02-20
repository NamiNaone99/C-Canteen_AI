import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO

# Load an image
image_path = r"/mnt/c/Users/nongf/Desktop/CUNEX/IMG_3844.jpg"
image = cv2.imread(image_path)

# Create config
cfg = get_cfg()
cfg.merge_from_file(r"/mnt/c/Users/nongf/Desktop/CUNEX/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Adjust threshold if needed
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

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
human_indices = [i for i, cls in enumerate(pred_classes) if cls == 0]

# Extract only human data
human_boxes = pred_boxes[human_indices]
human_masks = pred_masks[human_indices]

# Draw segmentation masks & bounding boxes
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(instances[human_indices])

# Save the output image
output_path = "output_human_segmented.jpg"
cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Make prediction
    outputs = predictor(image)
    
    # Get predictions
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    
    # Filter only human (COCO class "person" = 0)
    human_indices = [i for i, cls in enumerate(pred_classes) if cls == 0]
    
    # Number of people detected
    num_people = len(human_indices)
    
    return JSONResponse(content={"number_of_people": num_people})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_AI:app", host="0.0.0.0", port=8000, reload=True)