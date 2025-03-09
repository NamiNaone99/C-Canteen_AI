import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
import yaml
import random
from datetime import datetime
app = FastAPI()

# Load configuration from YAML file
config_path = r"/mnt/c/Users/nongf/Desktop/CUNEX/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load an image
image_path = config['image_path']['image1']
image = cv2.imread(image_path)

# Create config
cfg = get_cfg()
cfg.merge_from_file(config['cfg_file']['cfg1'])
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Adjust threshold if needed
cfg.MODEL.WEIGHTS = config['detectron']['weight']

# Create predictor
predictor = DefaultPredictor(cfg)

@app.post("/predict")
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

@app.get("/get_table")
async def get_table():
    # Define the number of rows per column
    canteen_name = "icanteen"
    column_constraints = {
        "A": 12, "B": 12, "C": 6, "D": 5, "E": 4, "F": 3,
        "G": 7, "H": 5, "I": 10, "J": 7, "K": 2
    }

    # Total number of rows
    total_rows = 12
    total_cols = 11  # Columns from A to K
    
    # Initialize the table with empty values
    table_data = [["-" for _ in range(total_cols)] for _ in range(total_rows)]

    # Assign random values (0-2) according to constraints
    for col_index, (col_letter, max_rows) in enumerate(column_constraints.items()):
        for row in range(max_rows):
            table_data[row][col_index] = random.randint(0, 2)

    # Convert to JSON format with column names as keys, removing "-"
    json_result = {
        col_letter: [table_data[row][col_index] for row in range(column_constraints[col_letter])]
        for col_index, col_letter in enumerate(column_constraints.keys())
    }

    # Add canteen name to the result
    json_result["canteen_name"] = canteen_name
    
    # Add timestamp to the result
    json_result["timestamp"] = datetime.now().isoformat()

    print(json_result)
    return JSONResponse(content=json_result)



# Predefined canteen IDs
canteens = {
    "โรงอาหารวิศวะ": "ChIJO0JPydWe4jARBaNUvkc8qqU",
    "โรงอาหารเศรษฐศาสตร์": "ChIJz_D_b2Cf4jARIkQ7--r8C5E",
    "โรงอาหารอักษร": "ChIJa-NwE9Se4jAReDGetxNVJFE",
    "โรงอาหารวิทยา": "ChIJmaZ0QtWe4jARaczHLX88uS8",
    "โรงอาหารหอใน": "ChIJsYh0oiyZ4jARJdCngP-hA3w",
    "โรงอาหารรัฐศาสตร์": "ChIJB9XFeSqf4jARyGGPawxKziU",
    "โรงอาหารจุฬาพัฒน์ 14": "ChIJ6eiDHwCZ4jARDMo0zJ6Rtgo",
    "โรงอาหารนิเทศ , นิติ": "ChIJJTXYKiWf4jARsZAiaiFzMs0",
    "ซุ้มอาหารคณะบัญชี": "-",
    "ศูนย์อาหารคณะครุศาสตร์": "ChIJjdOD2SqZ4jAREtgYkPMsr0s",
    "ศูนย์อาหารทวีวงศ์ถวัลยศักดิ์": "ChIJneikLTaf4jARoEH_eiTeEGY",
    "โรงอาหารใต้ศูนย์หนังสือจุฬาฯ": "ChIJb6WqUdKe4jARGJbYcvJJ_-Q",
    "โรงอาหารหอกลาง": "ChIJh3kgQSuZ4jARhyg4QOxSXpI"
}

@app.get("/get_density")
async def get_density(id: str = Query(None, description="Canteen ID")):
    def generate_density_data(canteen_name, canteen_id):
        return {
            "density": random.randint(0, 200),
            "canteen_id": canteen_id,
            "canteen_name": canteen_name,
            "timestamp": datetime.now().isoformat()
        }

    if id:
        # Find the canteen with the given ID
        for canteen_name, canteen_id in canteens.items():
            if canteen_id == id:
                return JSONResponse(content=generate_density_data(canteen_name, canteen_id))
        return JSONResponse(content={"error": "Canteen ID not found"}, status_code=404)
    else:
        # Return all canteens with random densities
        data = [generate_density_data(name, canteen_id) for name, canteen_id in canteens.items()]
        return JSONResponse(content=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_AI:app", host="0.0.0.0", port=9925, reload=True)