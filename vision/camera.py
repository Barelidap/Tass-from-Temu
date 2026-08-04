import cv2


class Camera:
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
    ) -> None:
        self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(
                "Cant open webcamera "
                "Check camera permissions."
            )

        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height,
        )

    def read(self):
        success, frame = self.capture.read()

        if not success:
            raise RuntimeError(
                "Could not read a frame from the webcam."
            )

        return frame

    def release(self) -> None:
        self.capture.release()