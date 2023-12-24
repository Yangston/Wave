import threading
import win32con
import win32gui
import win32api
import pythoncom
import pyWinhook as pyHook
from PIL import Image, ImageWin
import time

# Set the image path
image_path = 'testing/3cm.png'  # Use a PNG image for transparency
new_path = "testing/3cm_trans.png"

x = 0
y = 0
opacity = 100

# Create Transparent Image
with Image.open(image_path) as im:
    im2 = im.copy()
    im2.putalpha(180)
    im.paste(im2, im)
    im.save(new_path)

# Variables to store the previous rectangle coordinates
prev_x, prev_y = 0, 0

# Function to draw a transparent image on the desktop


def draw_transparent_image(x, y, opacity):
    global prev_x, prev_y

    desktop_dc = win32gui.GetDC(0)

    # Clear the previous content
    win32gui.InvalidateRect(
        0, (prev_x - 50, prev_y - 50, prev_x + 50, prev_y + 50), True)

    # Load the image with transparency
    image = Image.open(image_path).convert("RGBA")

    # Set the image opacity
    image.putalpha(opacity)

    # Get image size
    image_width, image_height = image.size

    # Get the new size by dividing the original size by 2
    new_size = (image.width // 2, image.height // 2)

    # Resize the image
    shrunk_img = image.resize(new_size)

    shrunk_image_width, shrunk_img_height = shrunk_img.size

    # Assuming desktop_dc is your device context
    dib = ImageWin.Dib(shrunk_img)

    # Draw the transparent image on the desktop
    dib.draw(desktop_dc, (x - shrunk_img.width // 2, y - shrunk_img.height //
             2, x + shrunk_img.width // 2, y + shrunk_img.height // 2))

    win32gui.ReleaseDC(0, desktop_dc)

    # Update the previous coordinates
    prev_x, prev_y = x, y

# Callback function for mouse events


# Callback function for mouse events
def on_mouse_event(event):
    if event.Message == win32con.WM_MOUSEMOVE:
        global x, y
        x, y = event.Position
    return True

# Callback function for keyboard events


def on_keyboard_event(event):
    if event.Ascii == 49:  # Check if the pressed key is '1'
        global x, y, opacity
        draw_transparent_image(x, y, opacity)
    return True

# Function to set up the mouse and keyboard hooks


def set_hooks():
    # Set up mouse hook
    mouse_hook = pyHook.HookManager()
    mouse_hook.MouseAll = on_mouse_event
    mouse_hook.HookMouse()

    # Set up keyboard hook
    keyboard_hook = pyHook.HookManager()
    keyboard_hook.KeyDown = on_keyboard_event
    keyboard_hook.HookKeyboard()

    # Start the message loop for both hooks
    pythoncom.PumpMessages()


# Create a thread to set up the hooks
hooks_thread = threading.Thread(target=set_hooks)
hooks_thread.start()

# Wait for the hooks thread to start
time.sleep(1)

# Run the Tkinter event loop (this loop is necessary to keep the script running)
root = win32gui.GetDesktopWindow()
win32gui.PumpMessages()
