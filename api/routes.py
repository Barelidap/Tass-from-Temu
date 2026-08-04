from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from vision.video_stream import generate_frames


router = APIRouter()

templates = Jinja2Templates(directory="templates")


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