import cv2
import mediapipe as mp
import numpy as np
import re

# Initialize the MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Drawing variables
drawing_color = (0, 255, 0)  # Default green color
thickness = 5  # Thickness of the pen
eraser_thickness = 50
prev_x, prev_y = None, None
drawing = False

# Create a canvas for drawing
canvas = None

# Track the selected button
selected_button = None  # <-- Add here

# Taskbar dimensions
taskbar_width = 50
button_size = 50
button_spacing = 60  # Space between buttons

# Button positions will be initialized later once we have the frame height
buttons = {}


# Set up the OpenCV window in normal mode
cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Detection', 1280, 720)  # Optional, adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Initialize the canvas if not already done
    if canvas is None:
        canvas = np.ones_like(frame) * 255  # White background

    # Get frame dimensions
    h, w, _ = frame.shape

    # Initialize button positions after getting frame height
    if not buttons:
        buttons = {
            'red': (w - taskbar_width, 10, w - 10, 10 + button_size),
            'green': (w - taskbar_width, 10 + button_spacing, w - 10, 10 + button_size + button_spacing),
            'blue': (w - taskbar_width, 10 + 2 * button_spacing, w - 10, 10 + button_size + 2 * button_spacing),
            'yellow': (w - taskbar_width, 10 + 3 * button_spacing, w - 10, 10 + button_size + 3 * button_spacing),
            'purple': (w - taskbar_width, 10 + 4 * button_spacing, w - 10, 10 + button_size + 4 * button_spacing),
            'orange': (w - taskbar_width, 10 + 5 * button_spacing, w - 10, 10 + button_size + 5 * button_spacing),
            'eraser': (w - taskbar_width, 10 + 6 * button_spacing, w - 10, 10 + button_size + 6 * button_spacing),
            'clear': (w - taskbar_width, 10 + 7 * button_spacing, w - 10, 10 + button_size + 7 * button_spacing),
            'exit': (w - taskbar_width, 10 + 8 * button_spacing, w - 10, 10 + button_size + 8 * button_spacing)
        }

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmark position of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to pixel values
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Check if the hand is in a writing gesture (index finger and thumb close together)
            if abs(x - thumb_x) < 30 and abs(y - thumb_y) < 30:
                drawing = True
            else:
                drawing = False

            # If previous coordinates exist, draw on the canvas
            if prev_x is not None and prev_y is not None:
                if drawing:
                    if drawing_color == (0, 0, 0):  # Eraser
                        cv2.line(canvas, (prev_x, prev_y), (x, y), drawing_color, eraser_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), drawing_color, thickness)

            # Check if the index finger is in the taskbar area
            if x > w - taskbar_width:
                for button, (x1, y1, x2, y2) in buttons.items():
                    if x1 < x < x2 and y1 < y < y2:
                        selected_button = button  # Track the selected button
                        if button == 'red':
                            drawing_color = (0, 0, 255)
                            thickness = 5
                        elif button == 'green':
                            drawing_color = (0, 255, 0)
                            thickness = 5
                        elif button == 'blue':
                            drawing_color = (255, 0, 0)
                            thickness = 5
                        elif button == 'yellow':
                            drawing_color = (0, 255, 255)  # Yellow
                            thickness = 5
                        elif button == 'purple':
                            drawing_color = (128, 0, 128)  # Purple
                            thickness = 5
                        elif button == 'orange':
                            drawing_color = (0, 165, 255)  # Orange
                            thickness = 5
                        elif button == 'eraser':
                            drawing_color = (0, 0, 0)
                            thickness = eraser_thickness
                        elif button == 'clear':
                            canvas = np.ones_like(frame) * 255
                            prev_x, prev_y = None, None
                        elif button == 'exit':
                            cap.release()
                            cv2.destroyAllWindows()
                            exit(0)  # Exit the program

            else:
                prev_x, prev_y = x, y

            # Draw pointer
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
    else:
        drawing = False

    # Combine the frame and canvas
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Draw buttons on the frame
    for button, (x1, y1, x2, y2) in buttons.items():
        # Highlight the selected button
        if button == selected_button:
            cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 255), 3)  # White border highlight

        if button == 'red':
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 20, (0, 0, 255), -1)
        elif button == 'green':
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 20, (0, 255, 0), -1)
        elif button == 'blue':
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 20, (255, 0, 0), -1)
        elif button == 'yellow':
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 20, (0, 255, 255), -1)  # Yellow
        elif button == 'purple':
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 20, (128, 0, 128), -1)  # Purple
        elif button == 'orange':
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 20, (0, 165, 255), -1)  # Orange
        elif button == 'eraser':
            cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2 - 10, y2 - 10), (0, 0, 0), -1)
        elif button == 'clear':
            cv2.putText(frame, 'C', ((x1 + x2) // 2 - 10, (y1 + y2) // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
        elif button == 'exit':
            cv2.putText(frame, 'X', ((x1 + x2) // 2 - 10, (y1 + y2) // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
