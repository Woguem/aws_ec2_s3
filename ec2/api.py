from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import shutil

from classifier import train_model, load_model, predict
from s3_utils import upload_to_s3, download_from_s3

# Load the trained model
model = load_model()

# Define FastAPI app
app = FastAPI(title="Iris Classifier API")

# Request body schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"msg": "Iris classifier is up and running!"}

@app.post("/predict")
def predict_species(features: IrisFeatures):
    input_data = [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]
    prediction = predict(model, input_data)
    return {"prediction": prediction}

@app.post("/upload-data")
def upload_data(file: UploadFile = File(...)):
    file_location = "data.csv"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    upload_to_s3(file_location, "iris_data.csv")
    return {"status": "Uploaded to S3"}

@app.post("/retrain")
def retrain():
    download_from_s3("iris_data.csv", "latest.csv")
    global model
    model = train_model("latest.csv")
    return {"status": "Model retrained from latest data in S3"}
