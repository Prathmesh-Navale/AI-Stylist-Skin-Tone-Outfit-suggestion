import os
import random
import cv2
import numpy as np

# --- CONFIGURATION ---
# If False, we won't even try to load the heavy ML libraries
TRY_AI_MODELS = True 
_MODEL_PATH = "skin_tone_model.h5"
CLASSES = ["fair", "brown", "black"]

# --- GLOBAL VARS ---
_MODEL = None
mp_face_detection = None
USE_MEDIAPIPE = False

# --- 1. ROBUST IMPORT BLOCK ---
try:
    # A. Try Loading TensorFlow
    if TRY_AI_MODELS:
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.image import img_to_array
            TF_AVAILABLE = True
        except ImportError:
            print("⚠️ TensorFlow not found. AI prediction disabled.")
            TF_AVAILABLE = False
    
    # B. Try Loading MediaPipe (Safe Import)
    try:
        import mediapipe as mp
        # Explicitly check if 'solutions' exists to catch the exact error you had
        if hasattr(mp, 'solutions'):
            mp_face_detection = mp.solutions.face_detection
            USE_MEDIAPIPE = True
            print("✅ MediaPipe loaded successfully.")
        else:
            print("⚠️ MediaPipe installed, but 'solutions' missing. (Check for file naming conflicts)")
            USE_MEDIAPIPE = False
    except ImportError:
        print("⚠️ MediaPipe library not found.")
        USE_MEDIAPIPE = False
        
except Exception as e:
    print(f"❌ critical import error: {e}")
    TF_AVAILABLE = False
    USE_MEDIAPIPE = False

def _load_model_once():
    global _MODEL
    if _MODEL is None and TF_AVAILABLE:
        if os.path.exists(_MODEL_PATH):
            try:
                _MODEL = load_model(_MODEL_PATH)
                print("✅ AI Model loaded.")
            except:
                _MODEL = None
        else:
            print("⚠️ Model file not found. Using Math Fallback.")

def normalize_lighting(image):
    """Simple lighting correction."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return image

def detect_skin_tone():
    """
    Main function called by Flask.
    Features:
    1. Opens Camera
    2. Uses MediaPipe (if working) OR Static Box (fallback)
    3. Uses AI Model (if loaded) OR Brightness Math (fallback)
    """
    _load_model_once()
    
    print("📷 Opening Webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Windows fallback
        
    if not cap.isOpened():
        print("❌ Camera failed to open.")
        return "brown" # Default fallback

    face_crop = None
    
    # If MediaPipe is broken, we define a static center box
    center_box_color = (0, 255, 255) # Yellow

    # If MediaPipe works, initialize it
    face_detector = None
    if USE_MEDIAPIPE:
        face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    print("👉 PRESS 'q' TO CAPTURE.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display_frame = frame.copy()

        # --- LOGIC BRANCH: SMART DETECT vs SIMPLE BOX ---
        if USE_MEDIAPIPE and face_detector:
            # 1. SMART MODE
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    wb = int(bboxC.width * w)
                    hb = int(bboxC.height * h)
                    cv2.rectangle(display_frame, (x, y), (x + wb, y + hb), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face Detected (Press Q)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                cv2.putText(display_frame, "Looking for face...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            # 2. FALLBACK MODE (Static Box)
            # Draw a box in the middle of the screen
            box_size = 200
            x1 = (w // 2) - (box_size // 2)
            y1 = (h // 2) - (box_size // 2)
            x2 = x1 + box_size
            y2 = y1 + box_size
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), center_box_color, 2)
            cv2.putText(display_frame, "Place face in box & Press Q", (x1 - 20, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_box_color, 2)

        cv2.imshow("Skin Tone Detector", display_frame)

        # CAPTURE LOGIC
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if USE_MEDIAPIPE and face_detector and results.detections:
                # Crop Smart
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                wb = int(bboxC.width * w)
                hb = int(bboxC.height * h)
                # Safety padding
                x, y = max(0, x), max(0, y)
                face_crop = frame[y:y+hb, x:x+wb]
            else:
                # Crop Simple (Center)
                box_size = 200
                x1 = (w // 2) - (box_size // 2)
                y1 = (h // 2) - (box_size // 2)
                x1, y1 = max(0, x1), max(0, y1)
                face_crop = frame[y1:y1+box_size, x1:x1+box_size]
            break

    # CLEANUP
    cap.release()
    cv2.destroyAllWindows()
    if USE_MEDIAPIPE and face_detector:
        face_detector.close()

    if face_crop is None or face_crop.size == 0:
        return "brown"

    # --- ANALYSIS ---
    try:
        corrected = normalize_lighting(face_crop)

        # Plan A: AI Model
        if _MODEL is not None:
            img = cv2.resize(corrected, (150, 150))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = _MODEL.predict(arr)
            return CLASSES[preds.argmax(axis=1)[0]]

        # Plan B: Math Fallback
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        print(f"Calculated Brightness: {brightness}")
        
        if brightness > 140: return "fair"
        if brightness < 90: return "black"
        return "brown"

    except Exception as e:
        print(f"Analysis failed: {e}")
        return "brown"