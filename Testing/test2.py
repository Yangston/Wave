import cv2
import numpy as np
import pyautogui
import time
import multitouch  # Import the multitouch script

# Initialize touch injection
multitouch.initialize()

# Define screen size
screen_width, screen_height = pyautogui.size()

# Define the starting positions of two touch points
point1 = (screen_width // 3, screen_height // 2)
point2 = (2 * screen_width // 3, screen_height // 2)

# Define the initial distance between the two points
initial_distance = 200

# Create a VideoCapture object to capture the webcam feed (you may need to adjust the camera index)
cap = cv2.VideoCapture(0)

# Lower and upper bounds for skin color in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Initialize flags for fist detection and pinch gesture
fist_detected = False
pinch_ongoing = False
distance = initial_distance  # Initialize distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if cv2.isContourConvex(approx) and cv2.contourArea(approx) > 1000:
            if len(approx) > 4:
                # Detect a fist-like gesture
                fist_detected = True
                pinch_ongoing = True
                break
            else:
                fist_detected = False
                pinch_ongoing = False

    if pinch_ongoing:
        # Calculate a new distance for the pinch-to-zoom gesture
        factor = 0.9  # Adjust this factor for zooming speed
        new_distance = distance * factor
        # Adjust point2 based on new_distance
        point2 = (point1[0] + new_distance, point1[1])
        # Simulate a pinch-to-zoom-out gesture by adjusting the touch positions
        multitouch.updateTouchInfo(0, True, point1[0], point1[1], 10)
        multitouch.updateTouchInfo(1, True, point2[0], point2[1])
        distance = new_distance  # Update the distance for the next frame
    else:
        multitouch.updateTouchInfo(0, False)
        multitouch.updateTouchInfo(1, False)

    # Apply the simulated touches
    multitouch.applyTouches()

    cv2.imshow('Fist Gesture Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
