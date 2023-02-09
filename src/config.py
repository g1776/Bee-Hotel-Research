from types import SimpleNamespace
from typing import Tuple, Union
import pytesseract
import os


class MotionCapConfig:
    """
    video (str): Path to video file
    timestamp (bool, optional): Whether a timestamp is present in the video. Defaults to False.
    timestamp_rect (Union[Tuple[float, float, float, float], None], optional): Coordinates of the timestamp rectangle. Defaults to None.
    detection_rate (int, optional): Number of frames to detect motion between. Defaults to 2.
    motion_threshold (float, optional): Threshold for motion detection (higher threshold = more motion needed). Defaults to 10.
    min_contour_area (float, optional): Minimum area of a contour to be considered motion. Defaults to 200.
    max_contour_area (float, optional): Maximum area of a contour to be considered motion. Defaults to 1000.
    log (str | None, optional): Path to log file. Defaults to None.
    motion_granularity (int, optional): Minimum number of frames between logging motion. If None defaults to FPS (aka 1 second).
    show (bool, optional): Whether to show the video. Defaults to True
    contour_window_size (int, optional): the number of frames to check no contours for. Defaults to 10.
    """

    # defaults
    VIDEO: str = ""
    TESSERACT: str = ""
    TIMESTAMP: bool = False
    TIMESTAMP_RECT: Union[Tuple[float, float, float, float], None] = None
    DETECTION_RATE: int = 2
    MOTION_THRESHOLD: float = 10
    MIN_CONTOUR_AREA: float = 200
    MAX_CONTOUR_AREA: float = 1000
    MOTION_GRANULARITY: int | None = None
    SHOW: bool = True
    LOG: str | None = None
    CONTOURS_WINDOW_SIZE: int = 10

    def __init__(self):

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
                case "CONTOURS_WINDOW_SIZE":
                    self.CONTOURS_WINDOW_SIZE = int(value)
                case _:
                    pass

        # any special intialization
        pytesseract.pytesseract.tesseract_cmd = self.TESSERACT

    def __repr__(self):
        return f"MotionCapConfig({self.__dict__})"

    __str__ = __repr__
