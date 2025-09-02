import os
import cv2
import mediapipe as mp

TURKISH_ALPHABET = [
    "A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ",
    "J", "K", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"
]

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

data_size = 200  # Her harften bu kadar toplanacak

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

for harf in TURKISH_ALPHABET:
    class_dir = os.path.join(DATA_DIR, harf)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"👉 {harf} harfi için veri toplanıyor. Hazırsan Q'ya bas.")

    while True:
        _, frame = cap.read()
        cv2.putText(frame, f"{harf} için Q'ya bas", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    counter = 0
    while counter < data_size:
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
            counter += 1

        cv2.imshow("frame", frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
