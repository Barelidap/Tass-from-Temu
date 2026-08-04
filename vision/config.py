MODEL_PATH = "yolo11n.pt"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CONFIDENCE_THRESHOLD = 0.7
DISAPPEARANCE_TIMEOUT = 2.0

WINDOW_NAME = "Tass from Temu"

# Hugging Face age-range classification model.
AGE_MODEL_NAME = "nateraw/vit-age-classifier"

# Try age estimation only once every this many frames.
# This prevents running the age model continuously.
AGE_ESTIMATION_INTERVAL = 15

# Stop trying when age estimation repeatedly fails.
MAX_AGE_ESTIMATION_ATTEMPTS = 10

# Ignore very small person crops.
MIN_PERSON_CROP_WIDTH = 100
MIN_PERSON_CROP_HEIGHT = 150

# InsightFace model pack.
#
# We load only its face-detection component, not face recognition,
# age estimation, or identity embeddings.
FACE_MODEL_NAME = "buffalo_l"

# Input size used by the SCRFD face detector.
FACE_DETECTION_SIZE = (640, 640)

# Minimum acceptable face-detection confidence.
FACE_DETECTION_CONFIDENCE = 0.6

# Add some space around the detected face before passing it
# into the age-estimation model.
FACE_CROP_MARGIN = 0.15

# Ignore faces that are too small for useful age estimation.
MIN_FACE_WIDTH = 40
MIN_FACE_HEIGHT = 40