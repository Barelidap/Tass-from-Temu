from dataclasses import dataclass
import time


@dataclass
class Visitor:
    first_seen: float
    last_seen: float

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


class VisitorTracker:
    def __init__(
        self,
        disappearance_timeout: float,
    ) -> None:
        self.disappearance_timeout = disappearance_timeout

        self.active_visitors: dict[int, Visitor] = {}
        self.completed_visit_durations: list[float] = []

    def update_visible_visitors(
        self,
        track_ids: list[int],
        current_time: float | None = None,
    ) -> None:
        if current_time is None:
            current_time = time.monotonic()

        for track_id in track_ids:
            if track_id not in self.active_visitors:
                self.active_visitors[track_id] = Visitor(
                    first_seen=current_time,
                    last_seen=current_time,
                )
            else:
                self.active_visitors[
                    track_id
                ].last_seen = current_time

    def finish_missing_visitors(
        self,
        current_time: float | None = None,
    ) -> list[tuple[int, float]]:
        if current_time is None:
            current_time = time.monotonic()

        finished_visits: list[tuple[int, float]] = []
        finished_ids: list[int] = []

        for track_id, visitor in self.active_visitors.items():
            time_missing = current_time - visitor.last_seen

            if time_missing > self.disappearance_timeout:
                duration = visitor.duration

                self.completed_visit_durations.append(
                    duration
                )

                finished_visits.append(
                    (track_id, duration)
                )

                finished_ids.append(track_id)

        for track_id in finished_ids:
            del self.active_visitors[track_id]

        return finished_visits

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

    @property
    def active_visit_count(self) -> int:
        return len(self.active_visitors)

    @property
    def completed_visit_count(self) -> int:
        return len(self.completed_visit_durations)

    @property
    def average_completed_duration(self) -> float:
        if not self.completed_visit_durations:
            return 0.0

        return (
            sum(self.completed_visit_durations)
            / len(self.completed_visit_durations)
        )