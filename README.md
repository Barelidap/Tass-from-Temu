# TASS from Temu

A modular computer vision project that simulates the core ideas behind retail analytics systems. 

The project uses a webcam to detect and anonymously track visitors in real time, estimate demographic information, and collect analytics without identifying individuals.

## Project Pipeline



                    Webcam
                       │
                       ▼
             YOLO Person Detection
                       │
                       ▼
          ByteTrack Object Tracking
                       │
                       ▼
             VisitorTracker Manager
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      Existing Visitor?        New Visitor
          │                         │
          │                  Create Visitor
          │                         │
          └────────────┬────────────┘
                       ▼
             Age already known?
                │            │
              Yes            No
                │            ▼
                │     Crop Person Image
                │            │
                │            ▼
                │     Face Detection
                │            │
                │            ▼
                │      Crop Face
                │            │
                │            ▼
                │     Age Estimation
                │            │
                └────────────┘
                       │
                       ▼
            Update Visitor Statistics
                       │
                       ▼
                Render Dashboard
### Architecture
The project follows a modular architecture.

### Camera
Responsible only for capturing webcam frames.

### Detector
Runs YOLO to detect people.

### Tracker
Uses ByteTrack to assign temporary IDs and keep them consistent between frames.

### VisitorTracker
Maintains the lifecycle of each visitor.

Each visitor stores:
-first appearance
-last appearance
-visit duration
-estimated age range
-age confidence

No personally identifiable information is stored.

### Face Detector
Detects a face only inside the detected person's bounding box.

### Age Estimator
Predicts an age range from the detected face.

### Renderer
Draws all overlays including:
-bounding boxes
-visitor IDs
-visit duration
-demographic information
-live statistics







