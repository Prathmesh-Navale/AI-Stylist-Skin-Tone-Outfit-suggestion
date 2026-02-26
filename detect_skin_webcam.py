import os
import cv2
import numpy as np

# ---------------- CONFIG ----------------
TRY_AI_MODELS = True
MODEL_PATH = "skin_tone_model.h5"
CLASSES = ["fair", "brown", "black"]

MODEL = None
USE_MEDIAPIPE = False

# ---------------- SAFE IMPORTS ----------------
TF_AVAILABLE = False
try:
    if TRY_AI_MODELS:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import img_to_array
        TF_AVAILABLE = True
except:
    print("⚠️ TensorFlow not available")

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp_face = mp.solutions.face_detection
        USE_MEDIAPIPE = True
except:
    print("⚠️ MediaPipe not available")

# ---------------- LOAD MODEL ONCE ----------------
def load_model_once():
    global MODEL
    if MODEL is None and TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            MODEL = load_model(MODEL_PATH)
            print("✅ Model Loaded")
        except:
            MODEL = None

# ---------------- LIGHT NORMALIZATION ----------------
def normalize_lighting(img):
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(3.0, (8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    except:
        return img

# ---------------- MAIN FUNCTION ----------------
def detect_skin_tone():
    load_model_once()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return "brown"

    face_detector = mp_face.FaceDetection(0.6) if USE_MEDIAPIPE else None
    face_crop = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display = frame.copy()

        if face_detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                d = results.detections[0]
                b = d.location_data.relative_bounding_box
                x, y = int(b.xmin * w), int(b.ymin * h)
                bw, bh = int(b.width * w), int(b.height * h)
                cv2.rectangle(display, (x, y), (x+bw, y+bh), (0,255,0), 2)
        else:
            s = 200
            x, y = w//2 - s//2, h//2 - s//2
            cv2.rectangle(display, (x,y), (x+s,y+s), (0,255,255), 2)

        cv2.imshow("Skin Tone Detector", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if face_detector and results.detections:
                face_crop = frame[y:y+bh, x:x+bw]
            else:
                face_crop = frame[y:y+s, x:x+s]
            break

    cap.release()
    cv2.destroyAllWindows()
    if face_detector:
        face_detector.close()

    if face_crop is None or face_crop.size == 0:
        return "brown"

    face = normalize_lighting(face_crop)

    # -------- AI MODE --------
    if MODEL is not None:
        img = cv2.resize(face, (150,150))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        return CLASSES[MODEL.predict(arr).argmax()]

    # -------- MATH FALLBACK --------
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])

    if brightness > 140: return "fair"
    if brightness < 90: return "black"
    return "brown"
