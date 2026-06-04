import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Load model once
face_model = FaceAnalysis()
face_model.prepare(ctx_id=-1)


def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    faces = face_model.get(image)

    if len(faces) == 0:
        return None

    face = faces[0]

    return {
        "embedding": face.embedding,
        "bbox": face.bbox.tolist()
    }


def cosine_similarity(embedding1, embedding2):

    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )

    return float(similarity)


def verify_faces(image1_path, image2_path):

    face1 = get_embedding(image1_path)
    face2 = get_embedding(image2_path)

    if face1 is None:
        return {"error": "Face not detected in image 1"}

    if face2 is None:
        return {"error": "Face not detected in image 2"}

    similarity = cosine_similarity(
        face1["embedding"],
        face2["embedding"]
    )

    threshold = 0.6

    if similarity >= threshold:
        result = "same person"
    else:
        result = "different person"

    return {
        "verification": result,
        "similarity_score": similarity,
        "bbox1": face1["bbox"],
        "bbox2": face2["bbox"]
    }