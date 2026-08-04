# TASS from Temu

A modular computer vision project that recreates the core ideas behind modern retail analytics systems using open-source tools.

The application processes a live webcam feed to detect and anonymously track visitors in real time, estimate demographic information, and collect analytics **without identifying individuals**.

## Features

- Real-time person detection using **YOLO**
- Multi-object tracking with **ByteTrack**
- Anonymous visitor tracking
- Face detection inside the detected person region
- Age estimation (performed only once per visitor)
- Gender estimation (performed only once per visitor)
- Live visitor statistics
- SQLite persistence for completed visits
- Modular, easy-to-extend architecture

---

# Project Pipeline

```text
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
      Demographics already known?
           │                    │
         Yes                    No
           │                    ▼
           │            Crop Person Image
           │                    │
           │                    ▼
           │             Face Detection
           │                    │
           │                    ▼
           │              Crop Face
           │                    │
           │                    ▼
           │        Estimate Missing Data
           │          ┌──────────────┐
           │          │              │
           │          ▼              ▼
           │    Age Estimation   Gender Estimation
           │          │              │
           └──────────┴──────────────┘
                       │
                       ▼
            Update Visitor Statistics
                       │
                       ▼
                Render Dashboard
```

---

# How It Works

Every frame from the webcam follows the same processing pipeline:

1. **YOLO** detects every visible person.
2. **ByteTrack** assigns a persistent tracking ID to each detected visitor.
3. The **VisitorTracker** creates or updates the corresponding visitor.
4. If the visitor's age or gender is still unknown:
   - the person's image is cropped,
   - the largest face is detected,
   - the face is cropped,
   - only the missing demographic estimator(s) are executed.
5. The estimated information is stored inside the visitor object.
6. Future frames reuse the stored values instead of running the models again.
7. The renderer displays all live analytics on screen.
8. If a visitor disappears from the frame, the system waits **2 seconds** before considering the visit complete. This helps avoid ending visits due to brief tracking interruptions.
9. Once a visit is completed, its statistics are permanently stored in a local **SQLite** database.

This significantly reduces unnecessary computation since demographic models are executed only once per visitor while also allowing long-term analytics to be collected across multiple sessions.

This significantly reduces unnecessary computation since demographic models are executed only once per visitor.

---

# Architecture

The project follows a modular architecture where each component has a single responsibility.

## Camera

Captures frames from the webcam.

---

## Detector

Runs **YOLO** to detect people in each frame.

Output:

- person bounding boxes
- detection confidence

---

## Tracker

Uses **ByteTrack** to assign temporary IDs and keep them consistent across frames.

Each tracked person receives a unique ID that exists only during the current visit.

---

## VisitorTracker

Maintains the complete lifecycle of every visitor.

Each visitor stores:

- first appearance
- last appearance
- visit duration
- entry timestamp
- exit timestamp
- estimated age range
- age confidence
- estimated gender
- gender confidence

The tracker also ensures that demographic estimation is performed only when necessary.

A visitor is considered to have left only after being absent from the frame for **2 seconds**. At that point, the visit is finalized and passed to the SQLite database for permanent storage.

No personally identifiable information is stored.

---

## Face Detector

Searches for the largest visible face only inside the detected person's bounding box.

Running face detection only within the person crop makes the pipeline faster and reduces false detections.

---

## Age Estimator

Predicts an age range from the detected face using a Vision Transformer model.

The prediction is saved into the visitor object and is not recomputed once successfully estimated.

---

## Gender Estimator

Predicts the apparent gender from the detected face.

Like age estimation, gender estimation is executed only until a successful prediction is obtained, after which the result is stored for the remainder of the visit.

---

## Renderer

Draws the live visualization, including:

- person bounding boxes
- visitor IDs
- visit duration
- estimated age
- estimated gender
- live visitor statistics

---
# SQLite Persistence

Completed visits are stored in a local **SQLite** database, allowing analytics to persist even after the application is restarted.

Each completed visit contains:

- visitor tracking ID
- entry timestamp
- exit timestamp
- visit duration
- estimated age range
- age confidence
- estimated gender
- gender confidence

The database serves as the foundation for future analytics dashboards, including:

- visitors by hour
- visitors by day
- age distribution
- gender distribution
- average visit duration
- traffic trends over time

Only completed visits are stored. Active visitors remain in memory until they leave the scene.

