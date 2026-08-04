import time
import uuid
from datetime import datetime

from collections.abc import Generator

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
    GENDER_ESTIMATION_INTERVAL,
    GENDER_MODEL_NAME,
    MAX_AGE_ESTIMATION_ATTEMPTS,
    MAX_GENDER_ESTIMATION_ATTEMPTS,
    MIN_FACE_HEIGHT,
    MIN_FACE_WIDTH,
    MIN_PERSON_CROP_HEIGHT,
    MIN_PERSON_CROP_WIDTH,
    MODEL_PATH,
)
from vision.face_detector import FaceDetector
from vision.gender_estimator import GenderEstimator
from vision.renderer import draw_person, draw_statistics
from vision.visitor_tracker import VisitorTracker
from database.visit_repository import VisitRepository
from api.statistics_events import notify_statistics_updated

def generate_frames() -> Generator[bytes, None, None]:
    # Create one unique identifier for this application run.
    #
    # ByteTrack IDs may restart from 1 whenever the program restarts,
    # so session_id lets us distinguish between separate runs.
    session_id = str(uuid.uuid4())

    # Opens data/tass.db and creates the visits table if necessary.
    visit_repository = VisitRepository(
        database_path="data/tass.db"
    )

    print(f"Application session: {session_id}")
    print(
        f"Existing database visits: "
        f"{visit_repository.get_visit_count()}"
    )

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

    gender_estimator = GenderEstimator(
        model_name=GENDER_MODEL_NAME
    )

    frame_number = 0

    try:
        while True:
            frame = camera.read()

            # Monotonic time is used for duration calculations.
            current_time = time.monotonic()

            # Real local time is stored in SQLite.
            current_datetime = datetime.now().astimezone()

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
                    current_datetime=current_datetime,
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
                        visitor_tracker.should_attempt_age_estimation(
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

                    should_estimate_gender = (
                        visitor_tracker
                        .should_attempt_gender_estimation(
                            track_id=track_id,
                            frame_number=frame_number,
                            estimation_interval=(
                                GENDER_ESTIMATION_INTERVAL
                            ),
                            maximum_attempts=(
                                MAX_GENDER_ESTIMATION_ATTEMPTS
                            ),
                        )
                    )

                    should_estimate_demographics = (
                        should_estimate_age
                        or should_estimate_gender
                    )

                    if (
                        crop_is_large_enough
                        and should_estimate_demographics
                    ):
                        # Each missing demographic keeps its own
                        # attempt counter.
                        if should_estimate_age:
                            visitor_tracker.register_age_attempt(
                                track_id=track_id,
                                frame_number=frame_number,
                            )

                        if should_estimate_gender:
                            visitor_tracker.register_gender_attempt(
                                track_id=track_id,
                                frame_number=frame_number,
                            )

                        # Crop the person once.
                        person_crop = frame[
                            y1:y2,
                            x1:x2,
                        ]

                        # Detect the face once and reuse it for both
                        # demographic models.
                        face_crop = (
                            face_detector.detect_largest_face(
                                person_crop
                            )
                        )

                        if face_crop is not None:
                            if should_estimate_age:
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

                            if should_estimate_gender:
                                gender_prediction = (
                                    gender_estimator.estimate(
                                        face_crop
                                    )
                                )

                                if gender_prediction is not None:
                                    visitor_tracker.set_gender(
                                        track_id=track_id,
                                        gender=(
                                            gender_prediction.gender
                                        ),
                                        confidence=(
                                            gender_prediction.confidence
                                        ),
                                    )

                    duration = (
                        visitor_tracker.get_live_duration(
                            track_id=track_id,
                            current_time=current_time,
                        )
                    )

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
                        gender=visitor.gender,
                        gender_confidence=(
                            visitor.gender_confidence
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

                gender_text = (
                    visitor.gender
                    if visitor.gender is not None
                    else "unknown"
                )

                database_visit_id = visit_repository.save_visit(
                    session_id=session_id,
                    tracker_id=track_id,
                    visitor=visitor,
                )

                # The SQLite transaction has completed successfully.
                # Notify the webpage that fresh statistics are available.
                notify_statistics_updated()


                print(
                    f"Visit saved: database ID {database_visit_id}, "
                    f"tracker ID {track_id}, "
                    f"entered {visitor.entered_at.isoformat()}, "
                    f"left {visitor.left_at.isoformat()}, "
                    f"duration {visitor.duration:.1f} seconds, "
                    f"age {age_text}, "
                    f"gender {gender_text}"
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

            # Convert the processed OpenCV frame into a JPEG image.
            success, encoded_frame = cv2.imencode(
                ".jpg",
                annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

            if not success:
                continue

            # Each yielded block becomes one image in the MJPEG stream.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded_frame.tobytes()
                + b"\r\n"
            )

    finally:
        # Release resources even when an exception occurs.
        camera.release()
        face_detector.close()
        visit_repository.close()