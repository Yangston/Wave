import tkinter as tk
import win32con
from PIL import Image, ImageTk, ImageSequence
import ctypes


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte),
                ("BlendFlags", ctypes.c_byte),
                ("SourceConstantAlpha", ctypes.c_byte),
                ("AlphaFormat", ctypes.c_byte)]


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long)]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]


class TransparentImageWindow:
    def __init__(self, root, image_path):
        self.root = root
        root.title("Transparent Image Window")

        # Set the window to be transparent
        root.attributes("-alpha", 0.9)  # Completely transparent
        # Set white as the transparent color
        root.wm_attributes('-transparentcolor', 'white')

        # Remove window frame and borders
        root.overrideredirect(True)

        # Load the GIF image
        scale = 2
        gif = Image.open(image_path)
        gif_width = gif.width * scale
        gif_height = gif.height * scale
        self.frames = [ImageTk.PhotoImage(frame.resize(
            (int(frame.width * scale), int(frame.height * scale)), Image.Resampling.LANCZOS)) for frame in ImageSequence.Iterator(gif)]

        # Calculate position to center the window on the screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_position = (screen_width - gif_width) // 2
        y_position = (screen_height - gif_height) // 2

        # Set the window position
        root.geometry(f"+{x_position}+{y_position}")

        # Create a canvas and add the GIF frames to it
        self.canvas = tk.Canvas(root, width=gif_width,
                                height=gif_height, highlightthickness=0)
        self.canvas.pack()

        # Set the window as a layered window
        hwnd = self.get_hwnd()

        # Initialize the first frame
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frames[0])

        # Schedule the update of the window frames
        self.root.after(0, self.update_window, 0)

    def update_window(self, frame_index):
        # Display the current frame
        self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.frames[frame_index])

        # Update the window with the new frame
        hwnd = self.get_hwnd()
        hdc = ctypes.windll.user32.GetDC(0)

        # Specify BLENDFUNCTION parameters as integers in the range 0-255
        blend = BLENDFUNCTION()
        blend.BlendOp = 0
        blend.BlendFlags = 0
        blend.SourceConstantAlpha = 255  # Source Constant Alpha (0-255)
        blend.AlphaFormat = win32con.AC_SRC_ALPHA

        rect = RECT()
        rect.left = self.root.winfo_x()
        rect.top = self.root.winfo_y()
        rect.right = rect.left + self.root.winfo_width()
        rect.bottom = rect.top + self.root.winfo_height()

        ptSrc = POINT()
        ptDst = POINT()

        ctypes.windll.user32.UpdateLayeredWindow(hwnd, hdc, ctypes.byref(ptDst), ctypes.byref(rect),
                                                 hdc, ctypes.byref(ptSrc), ctypes.byref(blend), win32con.ULW_ALPHA)

        ctypes.windll.user32.ReleaseDC(0, hdc)

        # Schedule the update for the next frame
        frame_index = (frame_index + 1) % len(self.frames)
        self.root.after(50, self.update_window, frame_index)

    def get_hwnd(self):
        return self.root.winfo_id()

    def start_main_loop(self):
        self.root.mainloop()


def display_gif(image_path):
    duration = 1500
    root = tk.Tk()
    app = TransparentImageWindow(root, image_path)

    # Schedule a function to close the window after a specified duration
    root.after(duration, root.destroy)

    root.mainloop()


if __name__ == "__main__":
    display_gif("testing/kon.gif")
