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
    results = model.track(
        source=frame,
        persist = True, #### keps tracking info between consecutive frames
        tracker = "bytetrack.yaml", ## tracking algo
        classes=[0],
        conf=0.7,
        verbose = False,
    )

    result = results[0]

    annotated_frame = frame.copy()

    curr_people_count = 0

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy() ## box coordinates

        track_ids = result.boxes.id.int().cpu().tolist() ## temporary tracking ids from bytetrack


        curr_people_count = len(track_ids)

        ## draw every box
        for box, track_id in zip(boxes, track_ids):
            x1,y1,x2,y2 = map(int, box)

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Person ID: {track_id}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
   # Display the current number of tracked people.
    cv2.putText(
        annotated_frame,
        f"Currently visible: {curr_people_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated_frame,
        "Press Q to quit",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(
        "Tass from Temu - Version 2",
        annotated_frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()