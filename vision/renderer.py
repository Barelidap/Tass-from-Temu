import cv2


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)

    return f"{minutes:02d}:{seconds:02d}"

def draw_person(
    frame,
    box,
    track_id: int,
    duration: float,
    confidence: float,
    age_group: str | None,
    age_confidence: float | None,
    gender: str | None,
    gender_confidence: float | None,
) -> None:
    """
    Draw a tracked person, visit duration, and demographic estimates.
    """

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    tracking_text = (
        f"ID {track_id} | "
        f"{format_duration(duration)} | "
        f"{confidence:.0%}"
    )

    if age_group is None:
        age_text = "Age: estimating"
    else:
        age_text = f"Age: {age_group}"

        if age_confidence is not None:
            age_text += f" ({age_confidence:.0%})"

    if gender is None:
        gender_text = "Gender: estimating"
    else:
        gender_text = f"Gender: {gender}"

        if gender_confidence is not None:
            gender_text += f" ({gender_confidence:.0%})"

    cv2.putText(
        frame,
        tracking_text,
        (x1, max(y1 - 60, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        age_text,
        (x1, max(y1 - 35, 50)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        gender_text,
        (x1, max(y1 - 10, 75)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

def draw_statistics(
    frame,
    visible_count: int,
    active_count: int,
    completed_count: int,
    average_duration: float,
) -> None:
    statistics = [
        f"Currently visible: {visible_count}",
        f"Active visits: {active_count}",
        f"Completed visits: {completed_count}",
        (
            "Average completed visit: "
            f"{format_duration(average_duration)}"
        ),
    ]

    for index, text in enumerate(statistics):
        cv2.putText(
            frame,
            text,
            (20, 35 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )