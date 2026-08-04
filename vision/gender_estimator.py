import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from vision.demographics import GenderPrediction


class GenderEstimator:
    """
    Estimates a binary gender label from a cropped face image.

    This class does not store visitor state. It only runs inference
    and returns a GenderPrediction.
    """

    def __init__(self, model_name: str) -> None:
        # The model is downloaded automatically on the first run.
        self.classifier = pipeline(
            task="image-classification",
            model=model_name,
        )

    def estimate(
        self,
        face_crop: np.ndarray,
    ) -> GenderPrediction | None:
        """
        Estimate a gender label from one face crop.
        """

        if face_crop.size == 0:
            return None

        # Convert OpenCV BGR into RGB.
        face_rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB,
        )

        face_image = Image.fromarray(face_rgb)

        predictions = self.classifier(face_image)

        if not predictions:
            return None

        # Predictions are returned from highest to lowest confidence.
        best_prediction = predictions[0]

        label = str(best_prediction["label"]).lower()

        # Normalize labels so the rest of the application receives
        # consistent values.
        if "female" in label or label == "woman":
            gender = "female"
        elif "male" in label or label == "man":
            gender = "male"
        else:
            gender = label

        return GenderPrediction(
            gender=gender,
            confidence=float(best_prediction["score"]),
        )