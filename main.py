import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "couldnt open the webcam bitch"
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH , 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    success, frame = camera.read()

    if not success:
        print("cant read frame from webcaaaaam")
        break

    ## class 0 is a person 
    ## verbose=false stops unnecessary info

    results = model.predict(
        source=frame,
        classes=[0],
        conf=0.5,
        verbose = False,
    )

    result = results[0]

    people_count = len(result.boxes)

    annotated_frame = result.plot() ##box

    cv2.putText(
        annotated_frame,
        f"people on frome: {people_count}",
        (20,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated_frame,
        "q to quit",
        (20,85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("TASS from Aliexpress", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()