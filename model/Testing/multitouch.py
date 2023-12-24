from ctypes import *
from ctypes.wintypes import *

# Constants
# For touchMask
TOUCH_MASK_NONE = 0x00000000  # Default
TOUCH_MASK_CONTACTAREA = 0x00000001
TOUCH_MASK_ORIENTATION = 0x00000002
TOUCH_MASK_PRESSURE = 0x00000004
TOUCH_MASK_ALL = 0x00000007

# For touchFlag
TOUCH_FLAG_NONE = 0x00000000

# For pointerType
PT_POINTER = 0x00000001  # All
PT_TOUCH = 0x00000002
PT_PEN = 0x00000003
PT_MOUSE = 0x00000004

# For pointerFlags
POINTER_FLAG_NONE = 0x00000000  # Default
POINTER_FLAG_NEW = 0x00000001
POINTER_FLAG_INRANGE = 0x00000002
POINTER_FLAG_INCONTACT = 0x00000004
POINTER_FLAG_FIRSTBUTTON = 0x00000010
POINTER_FLAG_SECONDBUTTON = 0x00000020
POINTER_FLAG_THIRDBUTTON = 0x00000040
POINTER_FLAG_FOURTHBUTTON = 0x00000080
POINTER_FLAG_FIFTHBUTTON = 0x00000100
POINTER_FLAG_PRIMARY = 0x00002000
POINTER_FLAG_CONFIDENCE = 0x00004000
POINTER_FLAG_CANCELED = 0x00008000
POINTER_FLAG_DOWN = 0x00010000
POINTER_FLAG_UPDATE = 0x00020000
POINTER_FLAG_UP = 0x00040000
POINTER_FLAG_WHEEL = 0x00080000
POINTER_FLAG_HWHEEL = 0x00100000
POINTER_FLAG_CAPTURECHANGED = 0x00200000

# Structs Needed


class POINTER_INFO(Structure):
    _fields_ = [("pointerType", c_uint32),
                ("pointerId", c_uint32),
                ("frameId", c_uint32),
                ("pointerFlags", c_int),
                ("sourceDevice", HANDLE),
                ("hwndTarget", HWND),
                ("ptPixelLocation", POINT),
                ("ptHimetricLocation", POINT),
                ("ptPixelLocationRaw", POINT),
                ("ptHimetricLocationRaw", POINT),
                ("dwTime", DWORD),
                ("historyCount", c_uint32),
                ("inputData", c_int32),
                ("dwKeyStates", DWORD),
                ("PerformanceCount", c_uint64),
                ("ButtonChangeType", c_int)
                ]


class POINTER_TOUCH_INFO(Structure):
    _fields_ = [("pointerInfo", POINTER_INFO),
                ("touchFlags", c_int),
                ("touchMask", c_int),
                ("rcContact", RECT),
                ("rcContactRaw", RECT),
                ("orientation", c_uint32),
                ("pressure", c_uint32)]

# Initialize Pointer and Touch info


def initialize(maxtouches=10, dwmode=1):
    global touches
    touches = (POINTER_TOUCH_INFO * maxtouches)()

    for ind in range(maxtouches):
        pointerInfo = POINTER_INFO(pointerType=PT_TOUCH,
                                   pointerId=ind,
                                   ptPixelLocation=POINT(950, 540),
                                   pointerFlags=POINTER_FLAG_NEW)
        touchInfo = POINTER_TOUCH_INFO(pointerInfo=pointerInfo,
                                       touchFlags=TOUCH_FLAG_NONE,
                                       touchMask=TOUCH_MASK_ALL,
                                       rcContact=RECT(pointerInfo.ptPixelLocation.x-5,
                                                      pointerInfo.ptPixelLocation.y-5,
                                                      pointerInfo.ptPixelLocation.x+5,
                                                      pointerInfo.ptPixelLocation.y+5),
                                       orientation=90,
                                       pressure=32000)
        touches[ind] = touchInfo

    if (windll.user32.InitializeTouchInjection(len(touches), 1) != 0):
        print("Initialized Touch Injection")


ids_to_update = 0
currently_down = None


def updateTouchInfo(id, down, x=0, y=0, fingerRadius=1, orientation=90, pressure=32000):
    global currently_down, ids_to_update
    if currently_down is None or len(currently_down) != len(touches):
        currently_down = [False] * len(touches)

    if down:
        touches[id].pointerInfo.pointerFlags = (((POINTER_FLAG_DOWN) if not currently_down[id] else POINTER_FLAG_UPDATE) |
                                                POINTER_FLAG_INRANGE |
                                                POINTER_FLAG_INCONTACT)
        touches[id].orientation = orientation
        touches[id].pressure = pressure
        touches[id].pointerInfo.ptPixelLocation.x = int(x)  # Convert to int
        touches[id].pointerInfo.ptPixelLocation.y = int(y)  # Convert to int
        touches[id].rcContact.left = int(x - fingerRadius)  # Convert to int
        touches[id].rcContact.right = int(x + fingerRadius)  # Convert to int
        touches[id].rcContact.top = int(y - fingerRadius)  # Convert to int
        touches[id].rcContact.bottom = int(y + fingerRadius)  # Convert to int
    else:
        # if currently_down[id] else POINTER_FLAG_UPDATE #if currently_down[id] else POINTER_FLAG_UPDATE
        touches[id].pointerInfo.pointerFlags = POINTER_FLAG_UP
    ids_to_update += 1 if down or currently_down[id] else 0
    currently_down[id] = down


def applyTouches():
    global ids_to_update
    if ids_to_update > 0:
        if (windll.user32.InjectTouchInput(int(ids_to_update), byref(touches[0])) == 0):
            print("Failed trying to update", ids_to_update,
                  "points with Error:", FormatError())
    ids_to_update = 0


# Use like:
# import multitouch       # Call once
# multitouch.initialize() # Call once
# for i in range(10):     # Call every frame
#    if i < len(keypoints):
#        multitouch.updateTouchInfo(i, True,
#            int(keypoints[i].pt[0] * 3),
#            int(keypoints[i].pt[1] * 3),
#            int(keypoints[i].size  / 2))
#    else:
#        multitouch.updateTouchInfo(i, False)
# multitouch.applyTouches()
