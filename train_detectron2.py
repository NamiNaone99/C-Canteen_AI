import torch
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import os
import requests
import zipfile


dataset_url = "https://app.roboflow.com/ds/cxnSAc1zDk?key=4eKN2wUaVM"
dataset_path = "human-dataset.zip"
extract_path = "human-dataset"

if not os.path.exists(extract_path):
    print("Downloading dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(dataset_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset ready!")


dataset_name = "human-dataset"
register_coco_instances(
    dataset_name + "_train", {}, "human-dataset/train/_annotations.coco.json", "human-dataset/train")
register_coco_instances(
    dataset_name + "_val", {}, "human-dataset/valid/_annotations.coco.json", "human-dataset/valid")

# Create config
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = (dataset_name + "_train",)
cfg.DATASETS.TEST = (dataset_name + "_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = "model_final_2d9806.pkl"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000  # Adjust iterations as needed
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Adjust based on memory

# Initialize Trainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save trained model and config
os.makedirs("output", exist_ok=True)
torch.save(trainer.model.state_dict(), "output/detectron2_model.pth")
with open("output/detectron2_config.yaml", "w") as f:
    f.write(cfg.dump())  # Save configuration
print("Model saved!")