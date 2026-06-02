import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy
import tensorflow
import mediapipe
import cv2
# -------------------------------
# MediaPipe Initialization
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

# -------------------------------
# Gesture Cooldown
# -------------------------------
last_gesture = ""
last_action_time = 0
cooldown = 1.2  # seconds

# -------------------------------
# Helper Functions
# -------------------------------

def fingers_up(hand_landmarks):

    tips = [4, 8, 12, 16, 20]

    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def detect_gesture(hand_landmarks):

    fingers = fingers_up(hand_landmarks)

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    pinch_distance = distance(thumb_tip, index_tip)

    # -------------------------------
    # Gesture Conditions
    # -------------------------------

    # Fist = Stop/Mute
    if fingers == [0, 0, 0, 0, 0]:
        return "MUTE"

    # Open Palm = Play/Pause
    elif fingers == [1, 1, 1, 1, 1]:
        return "PLAY_PAUSE"

    # Index Finger Only = Volume Up
    elif fingers == [0, 1, 0, 0, 0]:
        return "VOLUME_UP"

    # Pinky Finger Only = Volume Down
    elif fingers == [0, 0, 0, 0, 1]:
        return "VOLUME_DOWN"

    # Pinch Gesture = Screenshot
    elif pinch_distance < 0.05:
        return "SCREENSHOT"

    return None


def execute_gesture(gesture):

    global last_gesture, last_action_time

    current_time = time.time()

    # Prevent repeated triggers
    if gesture == last_gesture and (current_time - last_action_time) < cooldown:
        return

    last_gesture = gesture
    last_action_time = current_time

    try:

        if gesture == "PLAY_PAUSE":
            pyautogui.press("space")
            print("Play/Pause")

        elif gesture == "VOLUME_UP":
            pyautogui.press("volumeup")
            print("Volume Up")

        elif gesture == "VOLUME_DOWN":
            pyautogui.press("volumedown")
            print("Volume Down")

        elif gesture == "MUTE":
            pyautogui.press("volumemute")
            print("Mute")

        elif gesture == "SCREENSHOT":
            screenshot = pyautogui.screenshot()
            filename = f"screenshot_{int(time.time())}.png"
            screenshot.save(filename)
            print(f"Screenshot Saved: {filename}")

    except Exception as e:
        print("Gesture Error:", e)


# -------------------------------
# Main Loop
# -------------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hands
    results = hands.process(rgb_frame)

    gesture_text = "No Gesture"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Detect gesture
            gesture = detect_gesture(hand_landmarks)

            if gesture:
                gesture_text = gesture
                execute_gesture(gesture)

    # Display Gesture Text
    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show Window
    cv2.imshow("Media Gesture Control", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()