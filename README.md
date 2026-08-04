# TASS from Temu

A modular computer vision project that recreates the core ideas behind modern retail analytics systems using open-source tools.

The application processes a live webcam feed to detect and anonymously track visitors in real time, estimate demographic information, and collect analytics.

<img width="843" height="589" alt="Screenshot 2026-08-05 at 4 56 39 AM" src="https://github.com/user-attachments/assets/182af845-f6d9-4ef2-bbca-6f1e91766ac5" />


## Features

- Real-time person detection using **YOLO**
- Multi-object tracking with **ByteTrack**
- Anonymous visitor tracking
- Face detection inside the detected person region
- Age estimation (performed only once per visitor)
- Gender estimation (performed only once per visitor)
- Live visitor statistics
- Live video streaming through a **FastAPI** web interface
- Live completed-visit statistics displayed beside the video stream
- Automatic dashboard updates whenever a new visit is saved to SQLite
- Real-time browser notifications using **Server-Sent Events**
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
             Render Annotated Frame
                       │
                       ▼
          FastAPI MJPEG Video Stream
                       │
                       ▼
                Web Browser
                       │
                       ▼                         ↑
       Is the Visitor Missing for 2 Seconds?     |
         │                              │        |
        Yes                            No        |
         ▼                              ▼        |
Finalize Completed Visit                  -------
             │
             ▼
   Save Visit to SQLite
             │
             ▼
Send Statistics Update Event
             │
             ▼
Refresh Web Dashboard Statistics
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
10. After the database insert is completed, the application sends a statistics update event through **Server-Sent Events**.
11. The browser receives the event and requests the latest statistics from the FastAPI API.
12. The statistics panel beside the video stream updates automatically without refreshing the page.
    
This significantly reduces unnecessary computation since demographic models are executed only once per visitor while also allowing long-term analytics to be collected across multiple sessions.

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
Draws the live visualization directly onto each processed frame before it is streamed through the FastAPI web interface:

- person bounding boxes
- visitor IDs
- visit duration
- estimated age
- estimated gender
- live visitor statistics

---

## Statistics Event System

The statistics event system connects the computer vision pipeline with the browser dashboard.

When a completed visit is successfully saved to SQLite:

1. The video-processing pipeline triggers a statistics update notification.
2. FastAPI sends the notification to connected browsers using **Server-Sent Events**.
3. The browser requests the latest aggregated statistics from the API.
4. The statistics panel updates automatically without reloading the page.

This event-driven approach avoids repeatedly polling the database when no changes have occurred.

The dashboard currently displays:

- total completed visits
- average visit duration
- male visitor count
- female visitor count
- unknown gender count
- visitor counts grouped by estimated age range
  
---

# FastAPI Web Interface

Instead of displaying frames in a native OpenCV window, the application serves the processed video stream through a lightweight **FastAPI** web server.

The browser connects to the server using an MJPEG stream, allowing the live annotated frames to be viewed from any device on the same network.

The web interface currently provides:

- live camera stream
- real-time object detection and tracking
- rendered visitor analytics
- responsive browser-based visualization
- completed-visit statistics displayed beside the video stream
- automatic dashboard updates after every successful SQLite insert
- an API endpoint for retrieving the latest aggregated statistics
- a Server-Sent Events stream for database update notifications

The video and statistics use separate communication channels. The annotated frames are delivered through the continuous MJPEG stream, while completed-visit statistics are retrieved through a JSON API. A Server-Sent Events connection notifies the browser whenever the SQLite database receives a new completed visit.

This architecture cleanly separates the computer vision pipeline from the presentation layer, making it easy to extend the project with future dashboards, REST APIs, or remote monitoring capabilities.



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

The database now provides statistics to the live web dashboard and also serves as the foundation for additional analytics, including:

- visitors by hour
- visitors by day
- age distribution
- gender distribution
- average visit duration
- traffic trends over time

Only completed visits are stored. Active visitors remain in memory until they leave the scene.
Whenever a completed visit is inserted into the database, the application notifies the connected web dashboard. The browser then retrieves the latest aggregated values and updates the displayed statistics automatically.
