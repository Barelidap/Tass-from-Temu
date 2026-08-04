from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    app = FastAPI(
        title="TASS Vision",
        description="Anonymous real-time retail analytics",
        version="1.0.0",
    )

    # Make files inside static/ available through /static.
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static",
    )

    # Register routes from api/routes.py.
    app.include_router(router)

    return app


app = create_app()