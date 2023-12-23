import cv2
import pyautogui

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Load the Haar cascade for hand detection
hand_cascade = cv2.CascadeClassifier('hand.xml')

if hand_cascade.empty():
    raise Exception(
        "Haar cascade file not loaded. Make sure to provide the correct path.")

hand_detected = False
frames_without_detection = 0
frames_required_for_scroll = 10

# Initial scroll direction
scroll_direction = 0

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for hand detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(hands) > 0:
        hand_detected = True
        frames_without_detection = 0
        # Assuming only one hand is detected; you can adapt this code for multiple hands
        (x, y, w, h) = hands[0]

        # Draw a border around the detected hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mark key points on the hand (e.g., fingertips)
        # For simplicity, this code adds circles at the corners of the bounding box
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Top-left corner
        cv2.circle(frame, (x + w, y), 5, (0, 0, 255), -1)  # Top-right corner
        cv2.circle(frame, (x, y + h), 5, (0, 0, 255), -1)  # Bottom-left corner
        # Bottom-right corner
        cv2.circle(frame, (x + w, y + h), 5, (0, 0, 255), -1)

    else:
        frames_without_detection += 1

    if hand_detected:
        if frames_without_detection <= frames_required_for_scroll:
            if scroll_direction == 0:
                scroll_direction = -1  # Scroll up
                cv2.putText(frame, "Scroll Up", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            scroll_direction = 0
            hand_detected = False

    if scroll_direction == -1:
        # Scroll up
        pyautogui.scroll(3)  # You can adjust the scroll amount as needed

    # Display the video feed
    cv2.imshow('Hand Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
