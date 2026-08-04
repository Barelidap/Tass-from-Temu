from dataclasses import dataclass
import time


@dataclass
class Visitor:
    """
    Temporary state belonging to one ByteTrack ID.
    """

    first_seen: float
    last_seen: float

    # These remain None until age estimation succeeds.
    age_group: str | None = None
    age_confidence: float | None = None

    # Used to avoid running age estimation every frame.
    age_estimation_attempts: int = 0
    last_age_attempt_frame: int | None = None

    @property
    def duration(self) -> float:
        """
        Duration up to the most recent frame in which the visitor
        was visible.
        """

        return self.last_seen - self.first_seen

    @property
    def has_age_estimate(self) -> bool:
        """
        True after a valid age prediction has been stored.
        """

        return self.age_group is not None


class VisitorTracker:
    def __init__(
        self,
        disappearance_timeout: float,
    ) -> None:
        self.disappearance_timeout = disappearance_timeout

        # Dictionary key = ByteTrack ID.
        self.active_visitors: dict[int, Visitor] = {}

        # Store completed Visitor objects, not only their durations.
        # This lets us access their age later when adding a database.
        self.completed_visitors: list[Visitor] = []

    def update_visible_visitors(
        self,
        track_ids: list[int],
        current_time: float | None = None,
    ) -> list[int]:
        """
        Create visitors for new IDs and update last_seen for
        existing IDs.

        Returns IDs that were created during this call.
        """

        if current_time is None:
            current_time = time.monotonic()

        new_track_ids: list[int] = []

        for track_id in track_ids:
            visitor = self.active_visitors.get(track_id)

            if visitor is None:
                self.active_visitors[track_id] = Visitor(
                    first_seen=current_time,
                    last_seen=current_time,
                )

                new_track_ids.append(track_id)
            else:
                visitor.last_seen = current_time

        return new_track_ids

    def get_visitor(
        self,
        track_id: int,
    ) -> Visitor | None:
        """
        Retrieve the Visitor associated with a ByteTrack ID.
        """

        return self.active_visitors.get(track_id)

    def should_attempt_age_estimation(
        self,
        track_id: int,
        frame_number: int,
        estimation_interval: int,
        maximum_attempts: int,
    ) -> bool:
        """
        Decide whether age estimation should run for this visitor.
        """

        visitor = self.active_visitors.get(track_id)

        if visitor is None:
            return False

        # Age is already known, so no more inference is needed.
        if visitor.has_age_estimate:
            return False

        # Stop repeatedly trying when no usable face can be found.
        if visitor.age_estimation_attempts >= maximum_attempts:
            return False

        # Allow the first attempt immediately.
        if visitor.last_age_attempt_frame is None:
            return True

        frames_since_last_attempt = (
            frame_number - visitor.last_age_attempt_frame
        )

        return frames_since_last_attempt >= estimation_interval

    def register_age_attempt(
        self,
        track_id: int,
        frame_number: int,
    ) -> None:
        """
        Record that face detection and age estimation were attempted.
        """

        visitor = self.active_visitors.get(track_id)

        if visitor is None:
            return

        visitor.age_estimation_attempts += 1
        visitor.last_age_attempt_frame = frame_number

    def set_age(
        self,
        track_id: int,
        age_group: str,
        confidence: float,
    ) -> None:
        """
        Save the first successful age prediction.

        Once stored, it is not overwritten.
        """

        visitor = self.active_visitors.get(track_id)

        if visitor is None or visitor.has_age_estimate:
            return

        visitor.age_group = age_group
        visitor.age_confidence = confidence

    def get_live_duration(
        self,
        track_id: int,
        current_time: float | None = None,
    ) -> float:
        if current_time is None:
            current_time = time.monotonic()

        visitor = self.active_visitors.get(track_id)

        if visitor is None:
            return 0.0

        return current_time - visitor.first_seen

    def finish_missing_visitors(
        self,
        current_time: float | None = None,
    ) -> list[tuple[int, Visitor]]:
        """
        Finish visits whose IDs have not been seen for longer than
        the disappearance timeout.
        """

        if current_time is None:
            current_time = time.monotonic()

        finished_visits: list[tuple[int, Visitor]] = []
        finished_ids: list[int] = []

        for track_id, visitor in self.active_visitors.items():
            missing_duration = current_time - visitor.last_seen

            if missing_duration > self.disappearance_timeout:
                self.completed_visitors.append(visitor)

                finished_visits.append(
                    (track_id, visitor)
                )

                finished_ids.append(track_id)

        # Remove entries only after dictionary iteration finishes.
        for track_id in finished_ids:
            del self.active_visitors[track_id]

        return finished_visits

    @property
    def active_visit_count(self) -> int:
        return len(self.active_visitors)

    @property
    def completed_visit_count(self) -> int:
        return len(self.completed_visitors)

    @property
    def average_completed_duration(self) -> float:
        if not self.completed_visitors:
            return 0.0

        total_duration = sum(
            visitor.duration
            for visitor in self.completed_visitors
        )

        return total_duration / len(self.completed_visitors)