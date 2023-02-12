import cv2
from typing import List, Tuple
from math import dist
import numpy as np
from .logging import log_it
from collections import namedtuple
from .text_detect import text_detect

RECT_NAMEDTUPLE = namedtuple("RECT_NAMEDTUPLE", "x1 x2 y1 y2")


def overlap(rec1, rec2) -> bool:
    """Check if two rectangles overlap.

    Args:
        rec1 (RECT_NAMEDTUPLE): First rectangle
        rec2 (RECT_NAMEDTUPLE): Second rectangle

    Returns:
        bool: True if the rectangles overlap, False otherwise
    """
    if (rec2.x2 > rec1.x1 and rec2.x2 < rec1.x2) or (rec2.x1 > rec1.x1 and rec2.x1 < rec1.x2):
        x_match = True
    else:
        x_match = False
    if (rec2.y2 > rec1.y1 and rec2.y2 < rec1.y2) or (rec2.y1 > rec1.y1 and rec2.y1 < rec1.y2):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False


def get_tube_hives_coords(frame, log=None) -> np.ndarray:
    """
    Get the coordinates of the tube hives in the frame. Returns a numpy array of the coordinates.
    """
    tube_hives = cv2.HoughCircles(
        image=frame,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=60,
    )

    tube_hives: np.ndarray = np.around(tube_hives).astype(int)[0]

    # deal with logging
    tube_hives_msg = "Coordinates of Tube Hives detected, along with associated Bee ID:\n"
    tube_hives_msg += "\n".join(
        [f"Bee ID={i} Tube Hive Coords: {circle[:2]}" for i, circle in enumerate(tube_hives)]
    )
    tube_hives_msg += "\n\n"
    print(tube_hives_msg)
    if log:
        log_it(log, tube_hives_msg)

    return tube_hives


def get_contour_center(c) -> Tuple[int, int]:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def get_bb_center(bb) -> Tuple[int, int]:
    return (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2


def find_closest_circle(circles, point, config) -> int | None:
    """
    Find the closest circle to a point. Returns the index of the circle in the list of circles. If too far away, returns None.
    """
    # detect all circles in the frame

    # find circle closest to middle of contour
    closest_circle_dist = None
    index_of_closest_circle = None
    for i, circle in enumerate(circles):
        circle_dist = dist(np.array(circle)[:2], point)

        # if the circle is too far away, ignore it
        if circle_dist > config.MAX_DISTANCE_FROM_TUBE:
            continue

        if closest_circle_dist is None or circle_dist < closest_circle_dist:
            closest_circle_dist = circle_dist
            index_of_closest_circle = i

    return index_of_closest_circle


def draw_assigned_contour_on_frame(assigned_contour, frame) -> None:

    # extract values from assigned contour
    contour = assigned_contour["contour"]
    closest_tube = assigned_contour["closest_tube"]
    (x, y, w, h) = cv2.boundingRect(contour)

    cv2.rectangle(
        img=frame,
        pt1=(x, y),
        pt2=(x + w, y + h),
        color=(0, 255, 0),
        thickness=2,
    )

    # draw circle around closest circle
    if closest_tube is not None:
        cv2.circle(
            img=frame,
            center=(closest_tube[0], closest_tube[1]),
            radius=30,
            color=(0, 0, 255),
            thickness=2,
        )


# def process_contours_window(contours_window: List[List[dict]], TOTAL_FRAMES, config) -> None:
#     """Process the contour window to determine when a bee leaves the frame.
#         The idea here is that when a bee finally disappears, it is at the end of its
#         flight/movement path and it has most disappeared from the frame, into the tube hive.
#         This makes sure we only process the final frame the bee is visible to determine its ID.

#     Args:
#         contours_window (List[List[dict]]): The list of contours for each frame in the window.

#     Returns:
#         None
#     """

#     if len(contours_window) == 0:
#         return None

#     # process the contours window to determine when a bee leaves the frame
#     contours_to_log = []
#     last_frame = contours_window[-1]
#     if last_frame is None or len(last_frame) == 0:
#         return None

#     # show the frame in the contours that are being logged
#     frame_to_show = last_frame[0]["frame"]

#     # we do this since when there are no contours, assigned_contour is None
#     contours_to_check = [
#         contour_info for contour_info in last_frame if contour_info["assigned_contour"] is not None
#     ]

#     for contour_info in contours_to_check:
#         (x, y, w, h) = cv2.boundingRect(contour_info["assigned_contour"]["contour"])
#         overlap_found = False
#         for older_frame in contours_window[: len(contours_window) - 1]:

#             # if we found an overlap, we don''t need to keep checking this contour against old frames
#             if overlap_found:
#                 break

#             for older_contour_info in older_frame:

#                 # ignore frames with no contours
#                 if older_contour_info["assigned_contour"] is None:
#                     continue

#                 (o_x, o_y, o_w, o_h) = cv2.boundingRect(
#                     older_contour_info["assigned_contour"]["contour"]
#                 )
#                 if overlap(
#                     RECT_NAMEDTUPLE(
#                         x,
#                         x + w,
#                         y,
#                         y + h,
#                     ),
#                     RECT_NAMEDTUPLE(
#                         o_x,
#                         o_x + o_w,
#                         o_y,
#                         o_y + o_h,
#                     ),
#                 ):
#                     overlap_found = True
#                     break

#         if not overlap_found:
#             contours_to_log.append(contour_info)

#     # log the contours that made it through the window, and draw them on the frame_to_show
#     for contour_info in contours_to_log:

#         # extract elements from contour info
#         frame_count = contour_info["frame_count"]
#         bee_id = contour_info["assigned_contour"]["bee_id"]
#         timestamp_text = contour_info["timestamp"]

#         # show image
#         draw_assigned_contour_on_frame(contour_info["assigned_contour"], frame_to_show)

#         log_msg = generate_log_message(frame_count, TOTAL_FRAMES, timestamp_text, bee_id)
#         print(log_msg)
#         if config.LOG:
#             log_it(config.LOG, log_msg)

#     if config.SHOW:
#         cv2.imshow("🐝🏨 motion detector", frame_to_show)


def preprocess_frame(frame, config):
    """Preprocess a frame for motion detection

    Args:
        frame (np.ndarray): The frame to preprocess
        config (MotionCapConfig): The configuration object

    Returns:
        blurred (np.ndarray): The preprocessed frame

    """
    rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

    # put black rectangle over timestamp if present
    if config.TIMESTAMP:
        if config.TIMESTAMP_RECT is None:
            raise ValueError("Timestamp rectangle coordinates not provided")
        rgb = cv2.rectangle(
            img=rgb,
            pt1=config.TIMESTAMP_RECT[:2],
            pt2=config.TIMESTAMP_RECT[2:],
            color=(0, 0, 0),
            thickness=-1,
        )

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

    return blurred


def detect_contours_of_motion(preprocessed, previous_frame, config):
    """Detect motion in a frame and return the contours that represent motion

    Args:
        preprocessed (np.ndarray): The preprocessed frame
        previous_frame (np.ndarray): The previous preprocessed frame
        config (MotionCapConfig): The configuration object

    Returns:
        contours (List[np.ndarray]): The contours that represent motion
        previous_frame (np.ndarray): The current preprocessed frame

    """

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=preprocessed)
    previous_frame = preprocessed

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((2, 2))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>motion_threshold / 255)
    thresh_frame = cv2.threshold(
        src=diff_frame, thresh=config.MOTION_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY
    )[1]

    # 6. Find contours
    contours, _ = cv2.findContours(
        image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, previous_frame


def filter_contours(contours, tube_hives, config) -> List[dict]:
    """Filter out contours using a variety of techniques to reduce False Positives

    Args:
        contours (List[np.ndarray]): The contours to filter
        tube_hives (List[Tuple[int, int]]): The coordinates of the tube hives
        config (MotionCapConfig): The configuration object

    Returns:
        filtered_contours (List[dict]): The filtered contours, with bee_id and closest_tube

    Example output:
        [
            {
                "contour": np.ndarray,
                "bb": (x, y, w, h),
                "bee_id": 1,
                "closest_tube": (x, y)
            },
    """

    filtered_contours = []

    # filter out contours that are too small or too large
    filtered_on_size = [
        c
        for c in contours
        if config.MIN_CONTOUR_AREA < cv2.contourArea(c) < config.MAX_CONTOUR_AREA
    ]

    # find the closest circle to the detected motion.
    # we associate a bee with its closest circle once disappearing, hence bee_id
    for c in filtered_on_size:
        middle = get_contour_center(c)
        bee_id = find_closest_circle(tube_hives, middle, config)

        # If the bee_id is None, that means that it was not assigned to any tube (too far away)
        if bee_id is None:
            continue

        closest_tube = tube_hives[bee_id]

        filtered_contours.append(
            {
                "contour": c,
                "bb": cv2.boundingRect(c),
                "bee_id": bee_id,
                "closest_tube": closest_tube,
            }
        )

    return filtered_contours


def build_contour_window_entry(assigned_contours, frame, frame_count, config) -> List[dict]:
    # initialize the contour window entry
    contour_window_entry: List[dict] = []

    # for each contour assigned to a Bee ID, extract the timestamp and add it to the contour window entry
    for assigned_contour in assigned_contours:

        timestamp_text = text_detect(
            frame[
                config.TIMESTAMP_RECT[1] : config.TIMESTAMP_RECT[3],
                config.TIMESTAMP_RECT[0] : config.TIMESTAMP_RECT[2],
            ]
        )

        contour_window_entry.append(
            {
                "frame_count": frame_count,
                "frame": frame,
                "assigned_contour": assigned_contour,
                "timestamp": timestamp_text,
            }
        )

    # if no contours were assigned, add a None entry to the contour window entry.
    # This is so we can still display the frame if SHOW=True, even if no contours were detected
    if len(contour_window_entry) == 0:
        contour_window_entry.append(
            {
                "frame_count": frame_count,
                "frame": frame,
                "assigned_contour": None,
                "timestamp": None,
            }
        )

    return contour_window_entry
