from fastapi import FastAPI, UploadFile, File
from train import verify_faces
import os

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Face Authentication API is running"
    }


@app.post("/verify")
async def verify(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):

    image1_path = "temp1.jpg"
    image2_path = "temp2.jpg"

    # Save uploaded images
    with open(image1_path, "wb") as f:
        f.write(await image1.read())

    with open(image2_path, "wb") as f:
        f.write(await image2.read())

    # Run face verification
    result = verify_faces(
        image1_path,
        image2_path
    )
    os.remove(image1_path)
    os.remove(image2_path)
    return result