from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

model = tf.keras.models.load_model("mnist_model.keras")

@app.get("/")
def home():
    return {"message": "MNIST API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = image.resize((28, 28))

    img = np.array(image)

    # Normalize
    img = 255 - img
    img = img / 255.0

    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)

    digit = int(np.argmax(prediction))

    return {"prediction": digit}