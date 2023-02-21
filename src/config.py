from types import SimpleNamespace
from typing import Tuple, Union
import pytesseract
import os


class MotionCapConfig:
    def __init__(self):

        print("Loading config from environment variables...")

        # defaults

        # Path to video file
        self.VIDEO: str = ""

        # Path to tesseract executable
        self.TESSERACT: str = ""

        # Whether a timestamp is present in the video
        self.TIMESTAMP: bool = True

        # Coordinates of the timestamp rectangle
        self.TIMESTAMP_RECT: Union[Tuple[float, float, float, float], None] = (210, 20, 510, 50)

        # Number of frames to detect motion between
        self.DETECTION_RATE: int = 2

        # Threshold for motion detection (higher threshold = more motion needed)
        self.MOTION_THRESHOLD: float = 10

        # Minimum area of a contour to be considered motion
        self.MIN_CONTOUR_AREA: float = 200

        # Maximum area of a contour to be considered motion
        self.MAX_CONTOUR_AREA: float = 1000

        # Minimum number of frames between logging motion. If None defaults to FPS
        self.MOTION_GRANULARITY: int | None = None

        # Whether to show the video
        self.SHOW: bool = True

        # Path to log file
        self.LOG: str | None = None

        # the number of frames to check no overlapping contours for
        self.CONTOUR_WINDOW_SIZE: int = 10

        # The number of frames to wait before starting motion detection and finding the tube hives
        self.BUFFER_FRAMES: int = 200

        # The maximum distance from any tube a contour can be without being dropped
        self.MAX_DISTANCE_FROM_TUBE: int = 20

        # iterate through environment variables and set them as class variables
        for key, value in os.environ.items():
            try:
                value = eval(value)
            except:
                pass

            match key:
                case "VIDEO":
                    self.VIDEO = value
                case "TESSERACT":
                    self.TESSERACT = value
                case "TIMESTAMP":
                    self.TIMESTAMP = value
                case "TIMESTAMP_RECT":
                    self.TIMESTAMP_RECT = value
                case "DETECTION_RATE":
                    self.DETECTION_RATE = value
                case "MOTION_THRESHOLD":
                    self.MOTION_THRESHOLD = value
                case "MIN_CONTOUR_AREA":
                    self.MIN_CONTOUR_AREA = value
                case "MAX_CONTOUR_AREA":
                    self.MAX_CONTOUR_AREA = value
                case "MOTION_GRANULARITY":
                    self.MOTION_GRANULARITY = value
                case "SHOW":
                    self.SHOW = value
                case "LOG":
                    self.LOG = value
                case "CONTOUR_WINDOW_SIZE":
                    self.CONTOUR_WINDOW_SIZE = int(value)
                    if self.CONTOUR_WINDOW_SIZE < 0:
                        raise ValueError("CONTOUR_WINDOW_SIZE must be greater than or equal to 0")
                    if self.CONTOUR_WINDOW_SIZE == 0:
                        print("CONTOUR_WINDOW_SIZE is 0, no overlapping contours will be checked.")
                case "BUFFER_FRAMES":
                    self.BUFFER_FRAMES = int(value)
                case "MAX_DISTANCE_FROM_TUBE":
                    self.MAX_DISTANCE_FROM_TUBE = int(value)
                case _:
                    pass

        # any special intialization
        pytesseract.pytesseract.tesseract_cmd = self.TESSERACT

    def __repr__(self):
        return f"MotionCapConfig({self.__dict__})"

    __str__ = __repr__
