import time

import cv2
from ultralytics import YOLO

from vision.age_estimator import AgeEstimator
from vision.camera import Camera
from vision.config import (
    AGE_ESTIMATION_INTERVAL,
    AGE_MODEL_NAME,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONFIDENCE_THRESHOLD,
    DISAPPEARANCE_TIMEOUT,
    FACE_CROP_MARGIN,
    FACE_DETECTION_CONFIDENCE,
    FACE_DETECTION_SIZE,
    FACE_MODEL_NAME,
    MAX_AGE_ESTIMATION_ATTEMPTS,
    MIN_FACE_HEIGHT,
    MIN_FACE_WIDTH,
    MIN_PERSON_CROP_HEIGHT,
    MIN_PERSON_CROP_WIDTH,
    MODEL_PATH,
    WINDOW_NAME,
)
from vision.face_detector import FaceDetector
from vision.renderer import (
    draw_person,
    draw_statistics,
)
from vision.visitor_tracker import VisitorTracker


def main() -> None:
    # YOLO detects full people.
    # ByteTrack is used through model.track() below.
    person_model = YOLO(MODEL_PATH)

    camera = Camera(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    visitor_tracker = VisitorTracker(
        disappearance_timeout=DISAPPEARANCE_TIMEOUT
    )

    # SCRFD detects a face inside each YOLO person crop.
    #
    # Only the detection component of InsightFace is loaded.
    face_detector = FaceDetector(
        model_name=FACE_MODEL_NAME,
        detection_size=FACE_DETECTION_SIZE,
        minimum_confidence=FACE_DETECTION_CONFIDENCE,
        crop_margin=FACE_CROP_MARGIN,
        minimum_face_width=MIN_FACE_WIDTH,
        minimum_face_height=MIN_FACE_HEIGHT,
    )

    # The ViT classifier estimates an age group from a face crop.
    age_estimator = AgeEstimator(
        model_name=AGE_MODEL_NAME
    )

    frame_number = 0

    try:
        while True:
            frame = camera.read()
            current_time = time.monotonic()
            frame_number += 1

            # Detect people and maintain temporary ByteTrack IDs.
            results = person_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # COCO class 0 means person
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            result = results[0]
            annotated_frame = frame.copy()

            track_ids: list[int] = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .tolist()
                )

                # New ByteTrack IDs create new Visitor objects.
                # Existing IDs only have last_seen updated.
                visitor_tracker.update_visible_visitors(
                    track_ids=track_ids,
                    current_time=current_time,
                )

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    x1, y1, x2, y2 = map(int, box)

                    # Ensure the bounding box stays inside the frame.
                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    visitor = visitor_tracker.get_visitor(
                        track_id
                    )

                    if visitor is None:
                        continue

                    person_width = x2 - x1
                    person_height = y2 - y1

                    crop_is_large_enough = (
                        person_width >= MIN_PERSON_CROP_WIDTH
                        and person_height >= MIN_PERSON_CROP_HEIGHT
                    )

                    should_estimate_age = (
                        visitor_tracker
                        .should_attempt_age_estimation(
                            track_id=track_id,
                            frame_number=frame_number,
                            estimation_interval=(
                                AGE_ESTIMATION_INTERVAL
                            ),
                            maximum_attempts=(
                                MAX_AGE_ESTIMATION_ATTEMPTS
                            ),
                        )
                    )

                    if (
                        crop_is_large_enough
                        and should_estimate_age
                    ):
                        # Record the attempt even when face detection fails.
                        visitor_tracker.register_age_attempt(
                            track_id=track_id,
                            frame_number=frame_number,
                        )

                        # First crop the complete person from the frame.
                        person_crop = frame[
                            y1:y2,
                            x1:x2,
                        ]

                        # Then find and crop the face inside that body crop.
                        face_crop = (
                            face_detector.detect_largest_face(
                                person_crop
                            )
                        )

                        if face_crop is not None:
                            age_prediction = (
                                age_estimator.estimate(
                                    face_crop
                                )
                            )

                            if age_prediction is not None:
                                visitor_tracker.set_age(
                                    track_id=track_id,
                                    age_group=(
                                        age_prediction.age_group
                                    ),
                                    confidence=(
                                        age_prediction.confidence
                                    ),
                                )

                    duration = (
                        visitor_tracker.get_live_duration(
                            track_id=track_id,
                            current_time=current_time,
                        )
                    )

                    # The Visitor object now contains the age fields.
                    draw_person(
                        frame=annotated_frame,
                        box=(x1, y1, x2, y2),
                        track_id=track_id,
                        duration=duration,
                        confidence=confidence,
                        age_group=visitor.age_group,
                        age_confidence=(
                            visitor.age_confidence
                        ),
                    )

            finished_visits = (
                visitor_tracker.finish_missing_visitors(
                    current_time=current_time
                )
            )

            for track_id, visitor in finished_visits:
                age_text = (
                    visitor.age_group
                    if visitor.age_group is not None
                    else "unknown"
                )

                print(
                    f"Visit completed: ID {track_id}, "
                    f"duration {visitor.duration:.1f} seconds, "
                    f"age {age_text}"
                )

            draw_statistics(
                frame=annotated_frame,
                visible_count=len(track_ids),
                active_count=(
                    visitor_tracker.active_visit_count
                ),
                completed_count=(
                    visitor_tracker.completed_visit_count
                ),
                average_duration=(
                    visitor_tracker
                    .average_completed_duration
                ),
            )

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Release resources even when an exception occurs.
        camera.release()
        face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()