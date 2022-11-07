# Bee Hotel Research

## Gregory Glatzer

## The Pennsylvania State University

<img src="https://media2.giphy.com/media/U6YxrKZ84AfppW48r4/giphy.gif" width="100" height="100">

## Usage

The entrypoint to the tool is the `src.main.py` file. This file contains the driver code for motion capture. It is designed to be used as
a commandline tool. A basic usage is:

```bash
  python -m src.main --video data/10pm.mp4 --log 10pm.txt -t --timestamp-rect 210, 20, 510, 50
```

Here the `--video` argument specifies the video file to be processed.
The `--log` argument specifies the name of the log file to use.
The `-t` argument specifies that there is a timestamp present in the video.
The `--timestamp-rect` argument specifies the bounding box of the timestamp in the video. The bounding box is specified as `x1, y1, x2, y2`.

For more information on the commandline arguments, run:

```bash
  python -m src.main --help
```

To quit the running program, press `q`, or kill the terminal running the program.

### Alternative Usage: Config File (Recommended)

The `src.main.py` file can also be used with a config file. This is the recommended usage. The config file is a .env file that contains values for the commandline arguments. The config file is specified with the `--config` argument. For example:

```bash
  python -m src.main --config .env
```

The config file can be modified to change the commandline arguments. For example, to change the video file, change the `VIDEO` variable in the `.env` file.

## Methods

### Individual Bee Identification (IBI)

The concept behind individual bee identification (IBI) is based on the fact that solitary bees will claim a tube in the bee hotel. This means that the bee will return to the same tube every time it visits the bee hotel. This is a useful property because it allows us to identify individual bees. The IBI method is based on the following steps:

1. For each motion capture in the video, if the is not any more motion detected in that area of pixels (with a padding of 5 pixels) for 5 frames, then the bee is considered to havebe out of frame, and thus may have entered a tube.
2. After identifying that a bee has entered a tube, we must determine which tube it has entered. This is done by determining the closest tube to the bee's centroid. The closest tube is determined by the euclidean distance between the bee's centroid and the tube's centroid. Each tube is assigned a number, which in turn will be the bee's ID.

### Motion capture

Use OpenCV to detect motion in a video. This will be used to determine when a bee is present in the video. For a given input video log when the bee is present with a buffer of 5 second on each end.

Updates:

-   **10/15/2022**: Introduced _pytesseract_ to extract text from the video. This will be used to determine the timestamp that motion was captured. Before I tried to convert frame number to timestamp, but the frame rate appeared to be inconsistent. Currently the text is extracted correctly about 90% of the time. This can easily be solved. We just design the placement of the timestamp to be conducive to text extraction - Have the Raspberry Pi place the timestamp out of the frame on a black background.

    I also noticed that the motion capture completely fails when there is wind. This shakes the entire image. This can be seen several times through the 12pm video. Perhaps some form of image stabilization is needed.

-   **10/13/2022**: I have been able to create a basic implementation of motion detection using OpenCV. There are several parameters to tune in an attempt to reduce false positives. These variables are:

    -   detection_rate: Number of frames to detect motion between. Defaults to 2.
    -   motion_threshold: Threshold for motion detection (higher threshold = more motion needed). Defaults to 10.
    -   min_contour_area: Minimum area of a contour to be considered motion. Defaults to 200.

    I have also made the a parameter to choose how long to wait to log another motion event. In other words, if there are 20 events in a span of 2 seconds, and the wait time is 2 seconds, then all these events will be represented in the log as a single event. I call this "logging_granularity".
