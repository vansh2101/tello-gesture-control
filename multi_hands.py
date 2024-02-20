import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

#? Global Vars
labels = {0: 'forward', 1:'backward', 2:'up', 3:'down', 4:'left', 5:'right', 6:'flip', 7:'land'}
model = tf.keras.models.load_model('landmark-model.keras')

pred_left, pred_right = '', ''
temp_left, temp_right = '', ''
count_left, count_right = 0, 0

#? Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
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
    hands_num = []

    if res.multi_hand_landmarks:
        #? Drawing landmarks on the frame
        for id, handslms in enumerate(res.multi_hand_landmarks):
            draw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            hands_num.append(res.multi_handedness[id].classification[0].label)

            #? storing landmarks to feed the model
            pos = []
            for lm in handslms.landmark:
                pos.append(lm.x * x)
                pos.append(lm.y * y)

            landmarks.append(pos)

        #? Asking the model to predict the gesture
        prob = model.predict(landmarks)
        upper = 50

        for i, hand in zip(prob, hands_num):
            className = np.argmax(i)

            if i[className] > 0.9:
                # className = hand + ': ' + labels[className] + ' ' + str(round(i[className]*100, 2)) + '%'
                gesture = labels[className]

                if hand == 'Left':
                    if gesture == temp_left:
                        count_left += 1
                    else:
                        temp_left = gesture
                        count_left = 0

                    if count_left == 3:
                        pred_left = temp_left
                        count_left = 0

                    className = hand + ': ' + pred_left
                    
                
                elif hand == 'Right':
                    if gesture == temp_right:
                        count_right += 1
                    else:
                        temp_right = gesture
                        count_right = 0

                    if count_right == 3:
                        pred_right = temp_right
                        count_right = 0

                    className = hand + ': ' + pred_right
            else:
                className = ''

            cv2.putText(frame, className, (10, upper), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            upper += 50

    #? Show the frame
    cv2.imshow('Video Output', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()