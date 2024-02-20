import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

#? Global Vars
labels = {0: 'forward', 1:'backward', 2:'up', 3:'down', 4:'left', 5:'right', 6:'flip', 7:'land'}
model = tf.keras.models.load_model('landmark-model.keras')

#? Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

#? Initial the camera
vid = cv2.VideoCapture(0)

while True:
    _, frame = vid.read()

    frame = cv2.flip(frame, 1)
    y,x,n = frame.shape

    #? Preparing the image for hand detection
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb_img)

    landmarks = []

    if res.multi_hand_landmarks:
        #? Drawing landmarks on the frame
        for handslms in res.multi_hand_landmarks:
            draw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            
            #? storing landmarks to feed the model
            for lm in handslms.landmark:
                landmarks.append(lm.x * x)
                landmarks.append(lm.y * y)

        #? Asking the model to predict the gesture
        prob = model.predict(tf.expand_dims(landmarks, axis=0))
        className = np.argmax(prob)

        if prob[0][className] > 0.9:
            className = labels[className] + ' ' + str(round(prob[0][className]*100, 2)) + '%'
        else:
            className = ''

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    #? Show the frame
    cv2.imshow('Video Output', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()