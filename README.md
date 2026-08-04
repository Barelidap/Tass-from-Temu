# TASS from Temu

A modular computer vision project that simulates the core ideas behind retail analytics systems. 

The project uses a webcam to detect and anonymously track visitors in real time, estimate demographic information, and collect analytics without identifying individuals.

##Project Pipeline
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
  Existing Visitor?           New Visitor
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

