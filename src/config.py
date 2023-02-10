from types import SimpleNamespace
from typing import Tuple, Union
import pytesseract
import os


class MotionCapConfig:

    # defaults

    # Path to video file
    VIDEO: str = ""

    # Path to tesseract executable
    TESSERACT: str = ""

    # Whether a timestamp is present in the video
    TIMESTAMP: bool = True

    # Coordinates of the timestamp rectangle
    TIMESTAMP_RECT: Union[Tuple[float, float, float, float], None] = (210, 20, 510, 50)

    # Number of frames to detect motion between
    DETECTION_RATE: int = 2

    # Threshold for motion detection (higher threshold = more motion needed)
    MOTION_THRESHOLD: float = 10

    # Minimum area of a contour to be considered motion
    MIN_CONTOUR_AREA: float = 200

    # Maximum area of a contour to be considered motion
    MAX_CONTOUR_AREA: float = 1000

    # Minimum number of frames between logging motion. If None defaults to FPS
    MOTION_GRANULARITY: int | None = None

    # Whether to show the video
    SHOW: bool = True

    # Path to log file
    LOG: str | None = None

    # the number of frames to check no overlapping contours for
    CONTOUR_WINDOW_SIZE: int = 10

    # The number of frames to wait before starting motion detection and finding the tube hives
    BUFFER_FRAMES: int = 120

    # The maximum distance from any tube a contour can be without being dropped
    MAX_DISTANCE_FROM_TUBE: int = 20

    def __init__(self):

        print("Loading config from environment variables...")

        # iterate through environment variables and set them as class variables
        for key, value in os.environ.items():
            match key:
                case "VIDEO":
                    self.VIDEO = value
                case "TESSERACT":
                    self.TESSERACT = value
                case "TIMESTAMP":
                    self.TIMESTAMP = bool(value)
                case "TIMESTAMP_RECT":
                    self.TIMESTAMP_RECT = tuple(map(int, value.split(",")))
                case "DETECTION_RATE":
                    self.DETECTION_RATE = int(value)
                case "MOTION_THRESHOLD":
                    self.MOTION_THRESHOLD = float(value)
                case "MIN_CONTOUR_AREA":
                    self.MIN_CONTOUR_AREA = float(value)
                case "MAX_CONTOUR_AREA":
                    self.MAX_CONTOUR_AREA = float(value)
                case "MOTION_GRANULARITY":
                    self.MOTION_GRANULARITY = int(value)
                case "SHOW":
                    self.SHOW = bool(value)
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
