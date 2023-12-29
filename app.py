# Standard Library Imports
import itertools
import subprocess
import csv
import copy
import time as t
from collections import Counter
from collections import deque

# Third-Party Library Imports
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pyag
from playsound import playsound

# Local Imports
from utils import CvFpsCalc
from model import signDetector
from model import actionDetector


def main():
    debug = True
    pyag.FAILSAFE = False

    # Capture Variables _______________________________________________
    cap_device = 0
    cap_width = pyag.size()[0]
    cap_height = pyag.size()[1]
    screen_width, screen_height = pyag.size()
    use_static_image_mode = True

    # Detection Variables _____________________________________________
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
    num_hands = 2

    # Tensorflow Model Variables ______________________________________
    NUM_SIGNS = 15

    hand_sign_threshold = 0.5
    finger_gesture_threshold = 0.8
    hand_sign_confidence = 0
    finger_gesture_confidence = 0

    # Function Variables ______________________________________________
    # Point History
    histColour = (152, 251, 152)
    histSize = 2
    histBorder = 2

    cursor_position = (0, 0)

    # Scroll Variables
    MAX_SCROLL_SPEED = 250
    scroll_dist_threshold = 0.5
    scroll_direction = 0
    SCROLL_SENSITIVITY = 10  # Adjust sensitivity as needed
    prev_y = 0
    toScroll = False

    # Action Trigger Variables _______________________________________
    domainTriggered = False
    konTriggered = False
    click_triggered = False

    # Action Hold/Pause Variables ______________________________________
    hold = [0]*NUM_SIGNS
    hold_length = 5

    # Camera setup ______________________________________________________
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Mediapipe prep ____________________________________________________
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Model Initialization ____________________________________________
    keypoint_classifier = signDetector()
    point_history_classifier = actionDetector()

    # File Label Reading _____________________________________________
    signNames = 'model/signDetection/signNames.csv'
    actionNames = 'model/actionDetection/actionNames.csv'

    with open(signNames,
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(actionNames,
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement _________________________________________________
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Program Variables ________________________________________________
    # History
    recentSign = 0
    history_length = 32
    point_history = deque(maxlen=history_length)

    # Mode
    mode = 0
    signNameOffset = 10

    # Other
    use_brect = True

    # Sign Functions__________________________________________________
    def resetSigns():
        nonlocal histColour, histSize, histBorder, hold, domainTriggered, konTriggered
        histColour = (152, 251, 152)
        histSize = 1
        histBorder = 2
        hold = [0 for _ in hold]
        if handedness.classification[0].label[0:] == "Right":
            domainTriggered = False
            konTriggered = False

    def unknown():
        resetSigns()
        point_history.append([0, 0])

    def middleFinger():
        resetSigns()

    def openSign():
        resetSigns()
        point_history.append(landmark_list[8])

    def closeSign():
        resetSigns()

    def pointSign():
        nonlocal cursor_position
        alpha = 0.6  # Smoothing factor
        scaling = 1.5  # Scaling factor

        index_tip = hand_landmarks.landmark[8]  # Index fingertip landmark
        index_tip_x = int(index_tip.x * image.shape[1])
        index_tip_y = int(index_tip.y * image.shape[0])

        # Map the fingertip position to the screen coordinates
        screen_x = int(index_tip_x * screen_width / image.shape[1])
        screen_y = int(index_tip_y * screen_height / image.shape[0])

        # Apply EMA to smooth the cursor position
        new_x, new_y = screen_x, screen_y
        cursor_position = (
            int((1 - alpha) * cursor_position[0] + alpha * new_x),
            int((1 - alpha) * cursor_position[1] + alpha * new_y)
        )

        # Translate origin to center of screen
        cursor_position = (
            int(cursor_position[0] - screen_width/2),
            int(cursor_position[1] - screen_height/2)
        )

        # Scaling, preserve sign
        if (cursor_position[0] != 0 and cursor_position[1] != 0):
            cursor_position = (min(
                screen_width/2, abs(scaling*cursor_position[0]))*((scaling*cursor_position[0])/abs(scaling*cursor_position[0])), min(screen_height/2, abs(scaling*cursor_position[1]))*((scaling*cursor_position[1])/abs(scaling*cursor_position[1])))

        # Translate origin back to top left
        cursor_position = (
            int(cursor_position[0] + screen_width/2),
            int(cursor_position[1] + screen_height/2)
        )

        # Move the cursor to the fingertip position
        pyag.moveTo(cursor_position[0], cursor_position[1], _pause=False)

        point_history.append(landmark_list[8])
        resetSigns()

    def hollowPurpleSign():  # HollowPurple gesture
        nonlocal histColour, histSize, histBorder, hold
        if hold[hand_sign_id] > 0:
            resetSigns()
            point_history.append(landmark_list[8])
            histColour = (191, 0, 255)
            histSize = 4
            histBorder = 4
        hold[hand_sign_id] += 1

    def domainExpansionSign():  # DOMAIN EXPANSION
        nonlocal hold, domainTriggered
        if not domainTriggered:
            if hold[hand_sign_id] > hold_length:
                resetSigns()
                hold = [0 for _ in hold]
                playsound('domainExpansion.mp3', block=False)
                domainTriggered = True
            hold[hand_sign_id] += 1

    def peaceSign():
        nonlocal hold
        if hold[hand_sign_id] > hold_length:
            hold = [0 for _ in hold]
            resetSigns()
        hold[hand_sign_id] += 1

    def konSign():
        nonlocal hold, konTriggered
        if not konTriggered:
            if hold[hand_sign_id] > hold_length:
                hold = [0 for _ in hold]
                playsound('kon.mp3', block=False)
                subprocess.Popen(["python", "Testing/layeredWindows.py"])
                resetSigns()
                konTriggered = True
        hold[hand_sign_id] += 1

    def scrollHandDown():
        print(recentSign, " ", toScroll)
        if recentSign == hand_sign_index[scrollHandUp] and toScroll:
            scroll(scroll_direction, scroll_speed)
        resetSigns()

    def scrollHandUp():
        print(recentSign, " ", toScroll)
        if recentSign == hand_sign_index[scrollHandDown] and toScroll:
            scroll(scroll_direction, scroll_speed)
        resetSigns()

    def zoomFingersTgt():
        # if recentSign == hand_sign_index[zoomFingersApart]:
        # zoomIn()
        resetSigns()

    def zoomFingersApart():
        # if recentSign == hand_sign_index[zoomFingersTgt]:
        # zoomOut()
        resetSigns()

    def clickDown():
        nonlocal click_triggered
        if not click_triggered:
            # Hold down the click when all other fingertips are close
            pyag.mouseDown()
            click_triggered = True
        pointSign()

    def clickUp():
        nonlocal click_triggered
        if click_triggered:
            # Release the click when fingers move apart
            pyag.mouseUp()
            click_triggered = False
        pointSign()

    # Sign Functions _____________________________________________
    hand_sign_functions = {
        0: unknown,
        1: middleFinger,
        2: openSign,
        3: closeSign,
        4: pointSign,
        5: hollowPurpleSign,
        6: domainExpansionSign,
        7: peaceSign,
        8: konSign,
        9: scrollHandDown,
        10: scrollHandUp,
        11: zoomFingersTgt,
        12: zoomFingersApart,
        13: clickDown,
        14: clickUp
    }

    hand_sign_index = {value: key for key,
                       value in hand_sign_functions.items()}

    # Action Functions _____________________________________________
    hand_action_functions = {
        0: unknown,
        1: middleFinger,
        2: openSign,
        3: closeSign,
        4: pointSign,
        5: hollowPurpleSign,
        6: domainExpansionSign,
        7: peaceSign,
        8: konSign,
        9: scrollHandDown,
        10: scrollHandUp,
        11: zoomFingersTgt
    }

    def scroll(direction, speed):
        nonlocal toScroll

        if toScroll:
            if direction != 0:
                for s in range(SCROLL_SENSITIVITY):
                    pyag.scroll(
                        int(direction*speed), _pause=False)
                    toScroll = False
                    t.sleep(0.01)

    while True:
        fps = cvFpsCalc.get()

        # Key Input Detection
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        number, mode = select_mode(key, mode)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display

        debug_image = copy.deepcopy(image)

        # Hand Landmark Detection
        frame_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                if debug:
                    brect = calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list, signNameOffset)

                # Hand Signs ______________________________________________________________
                # Hand sign classification
                hand_sign_id, hand_sign_confidence = keypoint_classifier(
                    pre_processed_landmark_list)

                hand_sign_id += 1

                if (hand_sign_confidence < hand_sign_threshold):
                    hand_sign_id = 0

                # Triggers hand sign functions
                hand_sign_functions[hand_sign_id]()
                recentSign = hand_sign_id
                # point_history.append(landmark_list[8])

                # Hand Actions______________________________________________________________
                # Hand action classification
                hand_action_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (32 * 2):
                    hand_action_id, finger_gesture_confidence = point_history_classifier(
                        pre_processed_point_history_list)

                hand_action_id += 1

                if (finger_gesture_confidence < finger_gesture_threshold):
                    hand_action_id = 0

                # Triggers action functions
                # hand_action_functions[hand_action_id]()

                # Scroll Detection __________________________________________
                index_tip = hand_landmarks.landmark[8]
                distance = index_tip.y - prev_y

                print(f"Distance: {distance}")

                if abs(distance) > scroll_dist_threshold:
                    toScroll = True
                    if index_tip.y > prev_y:
                        # Hand moved down, scroll up
                        scroll_direction = 1
                        cv.putText(image, "Scroll Up", (50, 50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Hand moved up, scroll down
                        scroll_direction = -1
                        cv.putText(image, "Scroll Down", (50, 50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    scroll_speed = int(abs(distance)/10)
                    scroll_speed = min(scroll_speed, MAX_SCROLL_SPEED)

                prev_y = index_tip.y

                # Debug image drawing _______________________________________
                if debug:
                    debug_image = draw_bounding_rect(
                        use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[hand_action_id],
                    )
        else:
            point_history.append([0, 0])

        if debug:
            debug_image = draw_point_history(
                debug_image, point_history, histColour, histSize, histBorder)
            debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n - Normal
        mode = 0
    if key == 107:  # k - Sign
        mode = 1
    if key == 104:  # h - Point History
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list, numberOffset):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/signDetection/signData.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number + numberOffset, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/actionDetection/actionData.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    def draw_finger(start, end):
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[start + 1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[start + 1]),
                (255, 255, 255), 2)

    fingers = [[2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
               [13, 14, 15, 16], [17, 18, 19, 20]]

    for finger in fingers:
        for i in range(len(finger) - 1):
            draw_finger(finger[i], finger[i + 1])

    palm_connections = [(0, 1), (1, 2), (2, 5), (5, 9),
                        (9, 13), (13, 17), (17, 0)]

    for connection in palm_connections:
        cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                (255, 255, 255), 2)

    for landmark in landmark_point:
        cv.circle(image, tuple(landmark), 5, (255, 255, 255), -1)
        cv.circle(image, tuple(landmark), 5, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history, pointColour, pointSize, pointBorder):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2)*pointSize,
                      pointColour, pointBorder)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
