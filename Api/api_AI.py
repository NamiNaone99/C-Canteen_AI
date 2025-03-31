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
from typing import List
from pydantic import BaseModel

app = FastAPI()

# Load configuration from YAML file
config_path = r"../config.yaml"

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load an image



mode = "random"
@app.get("/update_mode/{new_mode}")
async def update_mode(new_mode: str):
    global mode
    if new_mode in ["random", "fixed", "fixed2"]:
        mode = new_mode
        return {"message": f"Mode updated to {mode}"}
    return {"error": "Invalid mode. Use 'random' or 'fixed'."}

@app.get("/get_table")
async def get_table():
    # Define the number of rows per column
    canteen_name = "icanteen"
    column_constraints = {
        "A": 7, "B": 7, "C": 6, "D": 5, "E": 4, "F": 3,
        "G": 7, "H": 5, "I": 10, "J": 7, "K": 2
    }

    # Total number of rows and columns
    total_rows = 12
    total_cols = 11  # Columns from A to K
   
    # Initialize table data
    table_data = [["-" for _ in range(total_cols)] for _ in range(total_rows)]


    fixed_col = "I"
    fixed_row = 1 
    #  J3,I4,I5
    fixed_row = fixed_row - 1
    if mode == "random":
        # Assign random values (0-2) according to constraints
        for col_index, (col_letter, max_rows) in enumerate(column_constraints.items()):
            for row in range(max_rows):
                table_data[row][col_index] = random.randint(0, 2)
    # Convert to JSON format with column names as keys, removing "-"
    json_result = {
        col_letter: [table_data[row][col_index] for row in range(column_constraints[col_letter])]
        for col_index, col_letter in enumerate(column_constraints.keys())
    }

    # Add additional metadata
    json_result["canteen_name"] = canteen_name
    json_result["timestamp"] = datetime.now().isoformat()
    json_result["mode"] = mode  # Include the mode in the response

    print(json_result)
    return JSONResponse(content=json_result)

# Simulated canteen mapping
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

# Pydantic model for request body
class CanteenIDs(BaseModel):
    ids: List[str]

@app.post("/get_density")
async def get_density(data: CanteenIDs):
    def generate_density_data(canteen_name, canteen_id):
        return {
            "density": random.randint(0, 200),
            "canteen_id": canteen_id,
            "canteen_name": canteen_name,
            "timestamp": datetime.now().isoformat()
        }

    results = []
    for query_id in data.ids:
        found = False
        for canteen_name, canteen_id in canteens.items():
            if canteen_id == query_id:
                results.append(generate_density_data(canteen_name, canteen_id))
                found = True
                break
        if not found:
            results.append({"error": f"Canteen ID '{query_id}' not found"})
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_AI:app", host="0.0.0.0", port=9925, reload=True)
    