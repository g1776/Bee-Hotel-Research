# Bee Hotel Research

## Gregory Glatzer

## The Pennsylvania State University

<hr/>

## Methods

### Method 1: motion capture

Use OpenCV to detect motion in a video. This will be used to determine when a bee is present in the video. For a given input video log when the bee is present with a buffer of 5 second on each end.

Updates:

- **10/15/2022**: Introduced _pytesseract_ to extract text from the video. This will be used to determine the timestamp that motion was captured. Before I tried to convert frame number to timestamp, but the frame rate appeared to be inconsistent. Currently the text is extracted correctly about 90% of the time. This can easily be solved. We just design the placement of the timestamp to be conducive to text extraction - Have the Raspberry Pi place the timestamp out of the frame on a black background.

  I also noticed that the notion capture completely fails when there is wind. This shakes the entire image. This can be seen several times through the 12pm video. Perhaps some form of image stabilization is needed.

- **10/13/2022**: I have been able to create a basic implementation of motion detection using OpenCV. There are several parameters to tune in an attempt to reduce false positives. These variables are:

  - detection_rate: Number of frames to detect motion between. Defaults to 2.
  - motion_threshold: Threshold for motion detection (higher threshold = more motion needed). Defaults to 10.
  - min_contour_area: Minimum area of a contour to be considered motion. Defaults to 200.

  I have also made the a parameter to choose how long to wait to log another motion event. In other words, if there are 20 events in a span of 2 seconds, and the wait time is 2 seconds, then all these events will be represented in the log as a single event. I call this "logging_granularity".
