import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Detects faces inside cropped person images using InsightFace.

    This class only detects and crops faces. It does not perform
    face recognition, create identity embeddings, or estimate age.
    """

    def __init__(
        self,
        model_name: str,
        detection_size: tuple[int, int],
        minimum_confidence: float,
        crop_margin: float,
        minimum_face_width: int,
        minimum_face_height: int,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if crop_margin < 0:
            raise ValueError(
                "crop_margin cannot be negative."
            )

        self.minimum_confidence = minimum_confidence
        self.crop_margin = crop_margin
        self.minimum_face_width = minimum_face_width
        self.minimum_face_height = minimum_face_height

        # Load only InsightFace's detection model.
        #
        # This prevents the application from loading face-recognition
        # and demographic-analysis modules that we do not need.
        self.detector = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )

        # ctx_id=-1 means CPU execution.
        #
        # det_size is the image size used internally by the
        # SCRFD face detector.
        self.detector.prepare(
            ctx_id=-1,
            det_size=detection_size,
        )

    def detect_largest_face(
        self,
        person_crop: np.ndarray,
    ) -> np.ndarray | None:
        """
        Detect faces inside one person's bounding box.

        Returns the largest valid face crop or None when no
        suitable face is found.
        """

        if person_crop.size == 0:
            return None

        crop_height, crop_width = person_crop.shape[:2]

        # InsightFace expects an OpenCV-style BGR image, so no
        # BGR-to-RGB conversion is required here.
        detected_faces = self.detector.get(
            person_crop,
            max_num=5,
        )

        valid_faces: list[
            tuple[int, tuple[int, int, int, int]]
        ] = []

        for face in detected_faces:
            # Face bounding box format:
            #
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)

            detection_score = float(face.det_score)

            if detection_score < self.minimum_confidence:
                continue

            # Keep coordinates within the person crop.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(crop_width, x2)
            y2 = min(crop_height, y2)

            face_width = x2 - x1
            face_height = y2 - y1

            if (
                face_width < self.minimum_face_width
                or face_height < self.minimum_face_height
            ):
                continue

            face_area = face_width * face_height

            valid_faces.append(
                (
                    face_area,
                    (x1, y1, x2, y2),
                )
            )

        if not valid_faces:
            return None

        # Usually there should be one face inside a person box.
        # If several are detected, choose the largest.
        _, largest_box = max(
            valid_faces,
            key=lambda candidate: candidate[0],
        )

        x1, y1, x2, y2 = largest_box

        face_width = x2 - x1
        face_height = y2 - y1

        # Add a margin around the face so the age model receives
        # the complete head rather than a tightly clipped crop.
        margin_x = int(face_width * self.crop_margin)
        margin_y = int(face_height * self.crop_margin)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)

        crop_x2 = min(
            crop_width,
            x2 + margin_x,
        )

        crop_y2 = min(
            crop_height,
            y2 + margin_y,
        )

        face_crop = person_crop[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if face_crop.size == 0:
            return None

        return face_crop

    def close(self) -> None:
        """
            artifact
        """

        pass