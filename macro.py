import pyautogui
import keyboard


def perform_macro():
    start = False

    # Simulate keypresses
    while True:
        if keyboard.is_pressed('s'):
            start = True
        if start:
            pyautogui.press('backspace')
            pyautogui.press('3')
            pyautogui.press('down')
        if keyboard.is_pressed('p'):
            start = False
        if keyboard.is_pressed('q'):
            print("Automation stopped.")
            break


if __name__ == "__main__":
    perform_macro()
