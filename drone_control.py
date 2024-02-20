import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from djitellopy import Tello


def tello_action(gesture):
    if gesture == 'forward':
        tello.send_rc_control(0,20,0,0)

    elif gesture == 'backward':
        tello.send_rc_control(0,-20,0,0)

    elif gesture == 'right':
        tello.send_rc_control(20,0,0,0)

    elif gesture == 'left':
        tello.send_rc_control(-20,0,0,0)

    elif gesture == 'up':
        tello.send_rc_control(0,0,20,0)

    elif gesture == 'down':
        tello.send_rc_control(0,0,-20,0)

    elif gesture == 'flip':
        tello.flip_forward()

    elif gesture == 'land':
        tello.land()
    else:
        pass


#? Global Vars
labels = {0: 'forward', 1:'backward', 2:'up', 3:'down', 4:'left', 5:'right', 6:'flip', 7:'land'}
model = tf.keras.models.load_model('landmark-model.keras')

pred = ''
temp = ''
count = 0

tello = Tello()
tello.connect()
print(tello.get_battery())

#? Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

#? Initial the camera
vid = cv2.VideoCapture(0)

inp = input('Do you want to takeoff? ')

if inp.lower() == 'y':
    tello.takeoff()

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
            gesture = labels[className]

            if gesture == temp:
                count += 1
            else:
                temp = gesture
                count = 0

            if count == 5:
                pred = temp
                count = 0

                tello_action(pred)

        cv2.putText(frame, pred, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Battery: '+str(tello.get_battery())+'%', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    #? Show the frame
    cv2.imshow('Video Output', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

tello.end()