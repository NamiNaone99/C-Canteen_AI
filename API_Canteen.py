from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config
from detectron2.structures import Instances

app = FastAPI()

# Load Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(r"/mnt/c/Users/nongf/Desktop/CUNEX/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Adjust threshold if needed
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

@app.post("/detect")
async def detect_people(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Run detection
    outputs = predictor(image)
    instances: Instances = outputs["instances"]
    
    # Count people (class ID for person in COCO dataset is 0)
    num_people = (instances.pred_classes == 0).sum().item()
    
    return {"num_people": num_people}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API_Canteen:app", host="0.0.0.0", port=8000, reload=True)
