import datetime
from typing import List, Tuple, Union
import cv2
import numpy as np
from .utils.motion_cap_helpers import *
from .utils.logging import *
from .utils.text_detect import text_detect
from src.config import MotionCapConfig


def motion_detector(
    config: MotionCapConfig,
):
    """Detect motion in a video"""

    print(config)

    # region init

    print(
        f"--- Motion detection session started at {datetime.datetime.now()} for file"
        f" {config.VIDEO} ---"
    )

    if config.LOG:
        init_logging_session(config.LOG, config.VIDEO)

    cap = cv2.VideoCapture(config.VIDEO)

    if config.MOTION_GRANULARITY is None:
        config.MOTION_GRANULARITY = int(cap.get(cv2.CAP_PROP_FPS))

    TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    previous_frame = None
    tube_hives = []

    # contours window is a list of contours information for the last config.CONTOURS_WINDOW_SIZE+1 frames
    # each element is a list of the contour information that happened in the associated frame.
    # Each contour information is a dictionary of {bee_id, frame, bb}. (bb = bounding box)
    # The last element in the list is the most recent frame.
    # The first element is the oldest frame, being CONTOURS_WINDOW_SIZE frames ago.
    contours_window: List[List[dict]] = []

    # endregion

    while True:

        # ignore the first 30 frames to allow the camera to adjust to the environment
        if frame_count < 30:
            frame_count += 1
            continue

        # log detected bee based on the processed contours window
        # This will happen CONTOURS_WINDOW_SIZE frames after the last time a bee was detected, since it needs to compare this many frames
        # the logging itself, however, will have the correct frame number of when the bee was detected
        contour_window_results = process_contours_window(contours_window)
        for contour_info in contour_window_results:

            # extract elements from contour info
            frame = contour_info["frame"]
            bee_id = contour_info["bee_id"]
            timestamp_text = contour_info["timestamp"]

            log_msg = generate_log_message(frame, TOTAL_FRAMES, timestamp_text, bee_id)
            print(log_msg)
            if config.LOG:
                log_it(config.LOG, log_msg)

        # region preprocess_frame

        # read image and convert to rgb
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

        # make a copy of the frame to draw on for visualization
        to_display = frame.copy()

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

        # endregion

        # wait 30 frames before detecting circles, in case it takes a couple seconds for the camera to focus
        if frame_count == 30:
            tube_hives = get_tube_hives_coords(blurred, config.LOG)

        # determine motion on every detection_rate-th frame
        frame_count += 1
        if (frame_count % config.DETECTION_RATE) == 0:

            # region detect_motion

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
                src=diff_frame, thresh=config.MOTION_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY
            )[1]

            # 6. Find contours
            contours, _ = cv2.findContours(
                image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )

            # endregion

            # filter out contours that are too small or too large
            contours_to_check = [
                c
                for c in contours
                if config.MIN_CONTOUR_AREA < cv2.contourArea(c) < config.MAX_CONTOUR_AREA
            ]

            # initialize the contour window entry
            contour_window_entry: List[dict] = []

            for contour in contours_to_check:

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
                # we associate a bee with its closest circle once disappearing, hence bee_id
                bee_id = find_closest_circle(tube_hives, middle)
                closest_circle = tube_hives[bee_id]

                # draw circle around closest circle
                if closest_circle is not None:
                    cv2.circle(
                        img=to_display,
                        center=(closest_circle[0], closest_circle[1]),
                        radius=30,
                        color=(0, 0, 255),
                        thickness=2,
                    )

                timestamp_text = text_detect(
                    frame[
                        config.TIMESTAMP_RECT[1] : config.TIMESTAMP_RECT[3],
                        config.TIMESTAMP_RECT[0] : config.TIMESTAMP_RECT[2],
                    ]
                )

                contour_window_entry.append(
                    {
                        "frame": frame_count,
                        "bb": (x, y, w, h),
                        "bee_id": bee_id,
                        "timestamp": timestamp_text,
                    }
                )

            # add to contours window, maintaining window size
            if len(contours_window) == config.CONTOURS_WINDOW_SIZE + 1:
                contours_window.pop(0)
            contours_window.append(contour_window_entry)

        # show image
        if config.SHOW:
            cv2.imshow("🐝🏨 motion detector", to_display)

        # check for quit operation
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting by user input.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {config.VIDEO}")
