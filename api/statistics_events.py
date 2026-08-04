import threading


# This condition lets the video-processing thread notify
# all connected statistics streams.
_statistics_condition = threading.Condition()

# The version increases every time SQLite receives a new visit.
_statistics_version = 0


def notify_statistics_updated() -> None:
    """
    Call this immediately after a completed visit is saved to SQLite.
    """

    global _statistics_version

    with _statistics_condition:
        _statistics_version += 1

        # Wake every browser currently waiting for an update.
        _statistics_condition.notify_all()


def get_statistics_version() -> int:
    """
    Return the current database-update version.
    """

    with _statistics_condition:
        return _statistics_version


def wait_for_statistics_update(previous_version: int) -> int:
    """
    Block until the statistics version changes.

    This function runs in a worker thread, so it does not block
    FastAPI's asyncio event loop.
    """

    with _statistics_condition:
        _statistics_condition.wait_for(
            lambda: _statistics_version != previous_version
        )

        return _statistics_version