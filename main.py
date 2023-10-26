import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize variables to track the finger position
index_base_x, index_base_y = 0, 0
screen_width, screen_height = pyautogui.size()

# Adjust this value based on your preferred distance threshold for a click
click_threshold = 0.10

# Define a threshold for finger proximity
# Adjust this value based on your preferred distance threshold for a click
finger_proximity_threshold = 70

fingertip_landmarks = [4, 8, 12, 16, 20]  # Landmark indices for fingertips

# Initialize a state variable to keep track of click
click_triggered = False

# Set the display window to full screen
cv2.namedWindow('Hand Tracking', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(
    'Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    # Set frame size to match screen size
    frame = cv2.resize(frame, (screen_width, screen_height))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for point in landmarks.landmark:
                x, y = int(point.x * frame.shape[1]
                           ), int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            index_base = landmarks.landmark[5]  # Index finger base
            index_base_x = int(index_base.x * frame.shape[1])
            index_base_y = int(index_base.y * frame.shape[0])

            index_tip = landmarks.landmark[8]  # Index fingertip landmark
            index_tip_x = int(index_tip.x * frame.shape[1])
            index_tip_y = int(index_tip.y * frame.shape[0])
            index_tip_depth = index_tip.z  # Depth value of index fingertip

            # Draw a line from the base of the index finger to the index finger tip
            cv2.line(frame, (index_base_x, index_base_y),
                     (index_tip_x, index_tip_y), (0, 0, 255), 2)

            # Map the fingertip position to the screen coordinates
            screen_x = int(index_tip_x * screen_width / frame.shape[1])
            screen_y = int(index_tip_y * screen_height / frame.shape[0])

            # Move the cursor to the fingertip position
            pyautogui.moveTo(screen_x, screen_y, _pause=False)

            fingertip_coordinates = []

            for idx in fingertip_landmarks:
                point = landmarks.landmark[idx]
                x, y, z = int(
                    point.x * frame.shape[1]), int(point.y * frame.shape[0]), point.z
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                fingertip_coordinates.append((x, y, z))

            # Calculate the distance between the thumb and middle and ring finger tips
            thumb_fingertip = fingertip_coordinates[0]
            other_fingertips = fingertip_coordinates[2:4]

            distances = [calculate_distance(
                thumb_fingertip[:2], fingertip[:2]) for fingertip in other_fingertips]

            # Check if all other fingertips are within the proximity threshold of the thumb
            all_fingertips_close = (distances[0] < finger_proximity_threshold)

            # If all other fingertips are close to the thumb, simulate a click
            if all_fingertips_close:
                if not click_triggered:
                    # Hold down the click when all other fingertips are close
                    pyautogui.mouseDown()
                    click_triggered = True
            else:
                if click_triggered:
                    # Release the click when fingers move apart
                    pyautogui.mouseUp()
                    click_triggered = False

            print(f"Index Tip Depth: {distances}")

            # if abs(index_tip_depth) > click_threshold:
            #     # Simulate a click when the index fingertip gets closer to the screen
            #     pyautogui.click()

            # # Print the depth value of the index fingertip
            # print(f"Index Tip Depth: {index_tip_depth}")

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
