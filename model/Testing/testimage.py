from PIL import Image, ImageWin
import win32gui
import win32con


def draw_transparent_image(image, x, y, opacity):
    hwnd = win32gui.GetDesktopWindow()
    hdc = win32gui.GetWindowDC(hwnd)

    width, height = image.size
    bmp = ImageWin.Dib(image)

    bmp.draw(hdc, (x - width // 2, y - height //
             2, x + width // 2, y + height // 2))

    # Set transparency using the Layered Windows API
    win32gui.UpdateLayeredWindow(hwnd, hdc, (x - width // 2, y - height // 2, x + width //
                                 2, y + height // 2), None, hdc, (0, 0), 0, win32con.AC_SRC_OVER, win32con.LWA_ALPHA)

    win32gui.ReleaseDC(hwnd, hdc)


image_path = 'testing/3cm.png'

image = Image.open(image_path).convert("RGBA")

# Get image size
image_width, image_height = image.size

# Get the new size by dividing the original size by 2
new_size = (image.width // 2, image.height // 2)

# Resize the image
shrunk_img = image.resize(new_size)

# Assuming shrunk_img is a PIL Image object
shrunk_img = shrunk_img.convert("RGBA")

# Set the opacity (alpha channel) for each pixel
# Set your desired opacity value (0-255, where 0 is fully transparent and 255 is fully opaque)
opacity = 128

# Get the alpha channel and update its values
alpha = shrunk_img.split()[3]
alpha = alpha.point(lambda i: i * (opacity / 255.0))

# Put the updated alpha channel back into the image
shrunk_img.putalpha(alpha)

# Specify the coordinates and call the draw_transparent_image function
x, y = 100, 100  # Adjust the coordinates as needed
draw_transparent_image(shrunk_img, x, y, opacity)
