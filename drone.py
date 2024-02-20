from djitellopy import Tello
import cv2
import threading
import keyboard


def handle_keys():
    global tello, taken_off

    def forward():
        tello.move_forward(40)

    def back():
        tello.move_back(40)

    def right():
        tello.move_right(40)

    def left():
        tello.move_left(40)

    def up():
        tello.move_up(40)

    def down():
        tello.move_down(40)

    def rotate():
        tello.rotate_clockwise(30)

    def flip():
        tello.flip_forward()

    def land():
        tello.land()

    if not taken_off:
        taken_off = True
        tello.takeoff()

    keyboard.add_hotkey('w', forward)
    keyboard.add_hotkey('s', back)
    keyboard.add_hotkey('a', left)
    keyboard.add_hotkey('d', right)
    keyboard.add_hotkey('e', rotate)
    keyboard.add_hotkey('f', flip)
    keyboard.add_hotkey('x', up)
    keyboard.add_hotkey('c', down)
    keyboard.add_hotkey('q', land)
    keyboard.wait()


def show_video():
    vid = cv2.VideoCapture('udp://0.0.0.0:11111')
    
    if not vid.isOpened():
        print("Error: Could not open video stream")
        tello.streamoff()
        tello.end()
        exit()

    while True:
        ret, frame = vid.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        print(tello.get_battery(),'%')

        cv2.imshow('Tello Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    tello.streamoff()
    tello.end()


tello = Tello()
tello.connect()
tello.streamon()

taken_off = False

thread1 = threading.Thread(target=show_video)
thread2 = threading.Thread(target=handle_keys)

thread1.start()
thread2.start()

# tello.takeoff()