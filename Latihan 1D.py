import pickle
import cv2
import mediapipe as mp
import numpy as np
import string

# Load model SIBI
model_dict = pickle.load(open('E:\Tugas-Tugas masuk sini titikduav\Praktikum Kontrol Cerdas\Pertemuan 3\model_sibi.p', 'rb'))
model = model_dict['model_sibi.p']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=1)

# Membuat dictionary 0-25 menjadi A-Z secara otomatis
labels_dict = {i: char for i, char in enumerate(string.ascii_uppercase)}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Prediksi huruf SIBI
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.putText(frame, f'Huruf: {predicted_character}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow('Deteksi SIBI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()