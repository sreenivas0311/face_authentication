# Face Authentication Using FastAPI and InsightFace

## Overview

This project is a simple face authentication (face verification) system built using Python, FastAPI, and InsightFace.

The application accepts two face images, extracts face embeddings using a pretrained InsightFace model, calculates cosine similarity, and determines whether the images belong to the same person.

---

## Features

* Face detection
* Face embedding extraction
* Cosine similarity calculation
* Same person / Different person prediction
* Bounding box extraction
* FastAPI endpoint for image upload and verification

---

## Project Structure

face_authentication/

* train.py
* test.py
* app.py
* images/
* requirements.txt
* README.md

---

## Technologies Used

* Python
* InsightFace
* ONNX Runtime
* OpenCV
* NumPy
* FastAPI
* Uvicorn

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Test File

```bash
python test.py
```

---

## Run FastAPI Application

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `/verify` endpoint to upload two images and check the verification result.

---

## Sample Response

```json
{
    "verification": "same person",
    "similarity_score": 0.83,
    "bbox1": [233.86, 122.75, 402.85, 344.12],
    "bbox2": [241.96, 218.19, 496.50, 567.56]
}
```

---

## Author

sreenivas
