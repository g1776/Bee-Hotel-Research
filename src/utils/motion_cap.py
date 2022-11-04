import datetime
from typing import Tuple, Union
import cv2
import numpy as np
from .motion_cap_helpers import *

from src.utils.text_detect import text_detect


def motion_detector(
    video,
    timestamp: bool = False,
    timestamp_rect: Union[Tuple[float, float, float, float], None] = None,
    detection_rate: int = 2,
    motion_threshold: float = 10,
    min_contour_area: float = 200,
    max_contour_area: float = 1000,
    log: str | None = None,
    motion_granularity: int = None,
    show: bool = True,
):
    """Detect motion in a video

    Args:
        video (str): Path to video file
        timestamp (bool, optional): Whether a timestamp is present in the vide. Defaults to False.
        timestamp_rect (Union[Tuple[float, float, float, float], None], optional): Coordinates of the timestamp rectangle. Defaults to None.
        detection_rate (int, optional): Number of frames to detect motion between. Defaults to 2.
        motion_threshold (float, optional): Threshold for motion detection (higher threshold = more motion needed). Defaults to 10.
        min_contour_area (float, optional): Minimum area of a contour to be considered motion. Defaults to 200.
        max_contour_area (float, optional): Maximum area of a contour to be considered motion. Defaults to 1000.
        log (str | None, optional): Path to log file. Defaults to None.
        motion_granularity (int, optional): Minimum number of frames between logging motion. If None defaults to FPS (aka 1 second).
        show (bool, optional): Whether to show the video. Defaults to True.

    """
    # based on https://towardsdatascience.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42

    print(
        f"--- Motion detection session started at {datetime.datetime.now()} for file {video} ---"
    )

    if log:
        init_logging_session(log, video)

    cap = cv2.VideoCapture(video)

    if motion_granularity is None:
        FPS = cap.get(cv2.CAP_PROP_FPS)
        motion_granularity = int(FPS)

    TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    previous_frame = None
    last_motion_frame = -1
    circles = []

    while True:

        # ignore the first 30 frames to allow the camera to adjust to the environment
        if frame_count < 30:
            frame_count += 1
            continue

        # read image and convert to rgb
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        # make a copy of the frame to draw on for visualization
        to_display = frame.copy()

        # put black rectangle over timestamp if present
        if timestamp:
            if timestamp_rect is None:
                raise ValueError("Timestamp rectangle coordinates not provided")
            rgb = cv2.rectangle(
                img=rgb,
                pt1=timestamp_rect[:2],
                pt2=timestamp_rect[2:],
                color=(0, 0, 0),
                thickness=-1,
            )

        # process image
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

        # wait 30 frames before detecting circles, in case it takes a couple seconds for the camera to focus
        if frame_count == 30:
            circles = cv2.HoughCircles(
                image=blurred,
                method=cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=60,
            )

            circles = np.around(circles).astype(int)[0]
            tube_hives_msg = "Coordinates of Tube Hives detected, along with associated Bee ID:\n"
            tube_hives_msg += "\n".join(
                [f"Bee ID={i} Tube Hive Coords: {circle[:2]}" for i, circle in enumerate(circles)]
            )
            tube_hives_msg += "\n\n"
            print(tube_hives_msg)
            if log:
                log_it(log, tube_hives_msg)

        # determine motion on every motion_n-th frame
        frame_count += 1
        if (frame_count % detection_rate) == 0:

            # 3. Set previous frame and continue if there is None
            if previous_frame is None:
                # First frame; there is no previous one yet
                previous_frame = blurred
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=blurred)
            previous_frame = blurred

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>motion_threshold / 255)
            thresh_frame = cv2.threshold(
                src=diff_frame, thresh=motion_threshold, maxval=255, type=cv2.THRESH_BINARY
            )[1]

            # 6. Find contours
            contours, _ = cv2.findContours(
                image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )

            motion_to_log = False
            for contour in contours:

                # If a contour is big enough, it's motion
                if (min_contour_area < cv2.contourArea(contour)) and (
                    max_contour_area > cv2.contourArea(contour)
                ):

                    # only log motion if there hasn't been any motion in the last "motion_granularity" frames
                    if frame_count - last_motion_frame > motion_granularity:
                        motion_to_log = True

                    # update last motion detection timestamp
                    last_motion_frame = frame_count

                    # draw rectangle around contour
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(
                        img=to_display,
                        pt1=(x, y),
                        pt2=(x + w, y + h),
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    # calculate middle of contour
                    middle = get_contour_center(contour)

                    # find the closest circle to the detected motion.
                    # we identify bees with which circle it was closest to, hence bee_id
                    bee_id = find_closest_circle(circles, middle)
                    closest_circle = circles[bee_id]

                    # draw circle around closest circle
                    if closest_circle is not None:
                        cv2.circle(
                            img=to_display,
                            center=(closest_circle[0], closest_circle[1]),
                            radius=30,
                            color=(0, 0, 255),
                            thickness=2,
                        )

            # log motion if we have a contour in motion and "motion_granularity" frames has passed since last motion
            if motion_to_log:

                timestamp = text_detect(
                    frame[
                        timestamp_rect[1] : timestamp_rect[3],
                        timestamp_rect[0] : timestamp_rect[2],
                    ]
                )
                log_msg = generate_log_message(frame_count, TOTAL_FRAMES, timestamp, bee_id)
                print(log_msg)
                if log:
                    log_it(log, log_msg)

        # show image
        if show:
            cv2.imshow("🐝🏨 motion detector", to_display)

        # check for quit operation
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting by user input.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {video}")
