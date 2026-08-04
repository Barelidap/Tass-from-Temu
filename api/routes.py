import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from api.statistics_events import (
    get_statistics_version,
    wait_for_statistics_update,
)
from database.visit_repository import VisitRepository
from vision.video_stream import generate_frames


router = APIRouter()

templates = Jinja2Templates(directory="templates")

DATABASE_PATH = "data/tass.db"


def read_statistics() -> dict:
    """
    Open a short-lived SQLite connection, read statistics,
    and close the connection.

    The video-processing code has its own repository connection.
    FastAPI uses a separate connection for reading.
    """

    repository = VisitRepository(database_path=DATABASE_PATH)

    try:
        return repository.get_statistics()
    finally:
        repository.close()


@router.get("/")
def home(request: Request):
    """
    Display the main TASS Vision webpage.
    """

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@router.get("/video-feed")
def video_feed():
    """
    Continuously send processed JPEG frames to the browser.
    """

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/statistics")
def statistics():
    """
    Return the current completed-visit statistics.

    The browser calls this once when the page first opens and again
    whenever the SSE stream announces a database update.
    """

    return read_statistics()


async def statistics_event_stream():
    """
    Keep one Server-Sent Events connection open.

    An event is sent only when video_stream.py saves a new visit.
    """

    current_version = get_statistics_version()

    while True:
        # wait_for_statistics_update() is a blocking threading function.
        # asyncio.to_thread prevents it from blocking FastAPI.
        current_version = await asyncio.to_thread(
            wait_for_statistics_update,
            current_version,
        )

        message = json.dumps(
            {
                "type": "statistics_updated",
                "version": current_version,
            }
        )

        yield f"data: {message}\n\n"


@router.get("/api/statistics-stream")
async def statistics_stream():
    """
    Notify the browser whenever SQLite receives a completed visit.
    """

    return StreamingResponse(
        statistics_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )