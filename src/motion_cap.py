import datetime
from typing import List, Callable
import cv2
from .utils.motion_cap_helpers import *
from .utils.logging import *
from src.config import MotionCapConfig


def motion_detector(
    config: MotionCapConfig,
    imshow_callback: Callable = None,
):
    """Detect motion in a video

    Args:
        config (MotionCapConfig): Configuration object
        imshow_callback (callable, optional): Callback function to display the image. Defaults to None. (this is used for the streamlit app)
    """

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
    BASE_FRAME = None

    # contours window is a list of contours information for the last config.CONTOUR_WINDOW_SIZE+1 frames
    # each element is a list of the contour information that happened in the associated frame.
    # Each contour information is a dictionary of {bee_id, frame, bb}. (bb = bounding box)
    # The last element in the list is the most recent frame.
    # The first element is the oldest frame, being CONTOUR_WINDOW_SIZE frames ago.
    contours_window: List[List[dict]] = []

    # endregion

    while True:

        # ignore the first `BUFFER_FRAMES` frames to allow the camera to adjust to the environment
        if frame_count < config.BUFFER_FRAMES:
            frame_count += 1
            continue

        # log detected bee based on the processed contours window
        # This will happen CONTOUR_WINDOW_SIZE frames after the last time a bee was detected, since it needs to compare this many frames
        # the logging itself, however, will have the correct frame number of when the bee was detected.
        # NOTE: This is where the logging and displaying of the image happens
        process_contours_window(contours_window, TOTAL_FRAMES, config, imshow_callback)

        # read and preprocess the frame
        success, frame = cap.read()
        if not success:
            break
        preprocessed = preprocess_frame(frame, config)

        # Once we have reached the `BUFFER_FRAMES`th frame, we grab the tube hive coordinates and assign them to Bee IDs
        if frame_count == config.BUFFER_FRAMES:
            tube_hives = get_tube_hives_coords(preprocessed, config.LOG)
            base_frame = preprocessed

        # determine motion on every `DETECTION_RATE-th frame
        frame_count += 1
        if (frame_count % config.DETECTION_RATE) == 0:
            # 3. Set previous frame and continue if there is None
            if previous_frame is None:
                # First frame; there is no previous one yet
                previous_frame = preprocessed
                continue

            # Detect motion and grab the contours that represent this motion
            contours, previous_frame = detect_contours_of_motion(
                preprocessed, previous_frame, base_frame, config
            )

            # filter out contours on size and distance to tubes.
            # This will also assign bee_ids to the contours
            # NOTE: This is where the bee_ids are assigned
            assigned_contours = filter_contours(contours, tube_hives, config)

            # build the contour window entry for this frame
            contour_window_entry = build_contour_window_entry(
                assigned_contours, frame, frame_count, config
            )

            # add to contours window, maintaining window size
            if len(contours_window) == config.CONTOUR_WINDOW_SIZE + 1:
                contours_window.pop(0)
            contours_window.append(contour_window_entry)

        # check for quit operation
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting by user input.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {config.VIDEO}")
