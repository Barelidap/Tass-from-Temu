import time

import cv2
from ultralytics import YOLO

from vision.camera import Camera
from vision.config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CONFIDENCE_THRESHOLD,
    DISAPPEARANCE_TIMEOUT,
    MODEL_PATH,
    WINDOW_NAME,
)
from vision.renderer import (
    draw_person,
    draw_statistics,
)
from vision.visitor_tracker import VisitorTracker


def main() -> None:
    model = YOLO(MODEL_PATH)

    camera = Camera(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    visitor_tracker = VisitorTracker(
        disappearance_timeout=DISAPPEARANCE_TIMEOUT
    )

    try:
        while True:
            frame = camera.read()
            current_time = time.monotonic()

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
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

                visitor_tracker.update_visible_visitors(
                    track_ids=track_ids,
                    current_time=current_time,
                )

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    duration = (
                        visitor_tracker.get_live_duration(
                            track_id=track_id,
                            current_time=current_time,
                        )
                    )

                    draw_person(
                        frame=annotated_frame,
                        box=box,
                        track_id=track_id,
                        duration=duration,
                        confidence=confidence,
                    )

            finished_visits = (
                visitor_tracker.finish_missing_visitors(
                    current_time=current_time
                )
            )

            for track_id, duration in finished_visits:
                print(
                    f"Visit completed: ID {track_id}, "
                    f"duration {duration:.1f} seconds"
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
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()