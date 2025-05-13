import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

emotion_model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

MY_NAME = "Иван"
MY_SURNAME = "Иванов"


def detect_emotion(face_roi):
    if emotion_model is None or face_roi.size == 0:
        return "Neutral"
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
    emotion_prediction = emotion_model.predict(reshaped_face)
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]
    return emotion_label

def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = 0

    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers += 1

    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers += 1
    return fingers

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_detection.process(rgb_frame)
        results_hand = hands.process(rgb_frame)

        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                              int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                is_my_face = True
                face_roi = frame[y:y+h, x:x+w]

                finger_count = 0
                if results_hand.multi_hand_landmarks:
                    for hand_landmarks in results_hand.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        finger_count = count_fingers(hand_landmarks)

                if is_my_face:
                    if finger_count == 1:
                        cv2.putText(frame, MY_NAME, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    elif finger_count == 2:
                        cv2.putText(frame, MY_SURNAME, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    elif finger_count == 3:
                        emotion = detect_emotion(face_roi)
                        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Неизвестный", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Face and Hand Detection', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()