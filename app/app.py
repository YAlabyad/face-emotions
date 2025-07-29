from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import os

# Load models
detection_model_path = os.getenv('HAARCASCAD_PATH')
emotion_model_path = os.getenv('MODEL_PATH')
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        faces = sorted(faces, reverse=True)[0]
        (fX, fY, fW, fH) = faces

        # Extract and preprocess ROI
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # Draw bounding box and label
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    return frame


def from_image_file(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: Could not read image at {img_path}")
        return

    processed = process_image(frame)
    cv2.imshow('Emotion Detection', processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_image(frame)
        cv2.imshow('Emotion Detection (Press Q to quit)', processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Main Program ---
print("Choose input method:")
print("1. Image file")
print("2. Webcam")
choice = input("Enter choice (1 or 2): ")

if choice == '1':
    img_path = input("Enter image path: ")
    from_image_file(img_path)
elif choice == '2':
    from_webcam()
else:
    print("Invalid choice")
