import threading
import win32con
import win32gui
import win32api
import pythoncom
import pyWinhook as pyHook
import time

# Set the red box size
box_size = 20

# Variables to store the previous rectangle coordinates
prev_x, prev_y = 0, 0

# Function to draw a red box on the desktop


def draw_red_box(x, y):
    global prev_x, prev_y

    desktop_dc = win32gui.GetDC(0)

    # Clear the previous content in the box area
    win32gui.InvalidateRect(0, (prev_x - box_size // 2, prev_y - box_size //
                            2, prev_x + box_size // 2, prev_y + box_size // 2), True)

    # Draw the new red box
    pen = win32gui.CreatePen(win32con.PS_SOLID, 2, win32api.RGB(255, 0, 0))
    win32gui.SelectObject(desktop_dc, pen)
    win32gui.Rectangle(desktop_dc, x - box_size // 2, y -
                       box_size // 2, x + box_size // 2, y + box_size // 2)

    win32gui.ReleaseDC(0, desktop_dc)

    # Update the previous coordinates
    prev_x, prev_y = x, y

# Callback function for mouse events


def on_mouse_event(event):
    if event.Message == win32con.WM_MOUSEMOVE:
        x, y = event.Position
        draw_red_box(x, y)
    return True

# Function to set up the mouse hook


def set_mouse_hook():
    hook = pyHook.HookManager()
    hook.MouseAll = on_mouse_event
    hook.HookMouse()

    pythoncom.PumpMessages()


# Create a thread to set up the mouse hook
hook_thread = threading.Thread(target=set_mouse_hook)
hook_thread.start()

# Wait for the hook thread to start
time.sleep(1)

# Run the Tkinter event loop (this loop is necessary to keep the script running)
root = win32gui.GetDesktopWindow()
win32gui.PumpMessages()
