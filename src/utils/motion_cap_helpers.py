import datetime
import cv2
from typing import Tuple
from math import dist
import numpy as np


def generate_log_message(frame: int, total_frames: int, timestamp: str, bee_id: int) -> str:
    return f"Bee ID={bee_id} detected at frame {frame}/{total_frames}, Timestamp: {timestamp}"


def log_it(fp: str, msg: str):
    with open(fp, "a") as f:
        f.writelines(f"{msg}\n")


def init_logging_session(fp, video_path):
    with open(fp, "a") as f:
        f.writelines(
            f"--- Logging session started at {datetime.datetime.now()} for file {video_path} ---\n"
        )


def get_contour_center(c) -> Tuple[int, int]:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def find_closest_circle(circles, point):
    """
    Find the closest circle to a point. Returns the index of the circle in the list of circles.
    """
    # detect all circles in the frame

    # find circle closest to middle of contour
    closest_circle = None
    closest_circle_dist = None
    index_of_closest_circle = None
    for i, circle in enumerate(circles):
        circle_dist = dist(np.array(circle)[:2], point)
        if closest_circle_dist is None or circle_dist < closest_circle_dist:
            closest_circle = circle
            closest_circle_dist = circle_dist
            index_of_closest_circle = i

    return index_of_closest_circle
