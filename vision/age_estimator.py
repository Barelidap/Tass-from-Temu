import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from vision.demographics import AgePrediction


class AgeEstimator:
    """
    Estimates an age range from a cropped face image.

    This class does not store visitor state. It only runs inference
    and returns an AgePrediction.
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
    ) -> AgePrediction | None:
        """
        Estimate an age range from one face crop.
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

        # The pipeline returns predictions ordered by confidence.
        best_prediction = predictions[0]

        return AgePrediction(
            age_group=str(best_prediction["label"]),
            confidence=float(best_prediction["score"]),
        )