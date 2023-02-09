from types import SimpleNamespace
from typing import Tuple, Union


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

    def __init__(self, config: SimpleNamespace):

        # iterate through config SipleNamespace and set attributes
        for key, value in config.__dict__.items():
            # capitalize key
            key = key.upper()
            setattr(self, key, value)

    def __repr__(self):
        return f"MotionCapConfig({self.__dict__})"

    __str__ = __repr__
