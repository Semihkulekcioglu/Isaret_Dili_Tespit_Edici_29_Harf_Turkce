import cv2
import mediapipe as mp
import numpy as np
import pickle

with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
    model = model_dict["model"]
    le = model_dict["label_encoder"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_prediction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        x_ = []
        y_ = []
        data_aux = []

        hand_landmarks = result.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append((lm.x - min(x_)) / (max(x_) - min(x_)))
            data_aux.append((lm.y - min(y_)) / (max(y_) - min(y_)))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_letter = le.inverse_transform(prediction)[0]
            prev_prediction = predicted_letter

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Tahmin: {prev_prediction}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("İşaret Dili Tanıma", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
