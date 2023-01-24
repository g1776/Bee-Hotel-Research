import cv2
from pytesseract import pytesseract
from datetime import datetime

INVALID_TIMESTAMP = "INVALID_TIMESTAMP"


def clean_timestamp(timestamp_to_parse: str) -> str:
    if len(timestamp_to_parse) < 5:
        # there just isn't enough to parse
        print("Dropping row with insufficient timestamp:", timestamp_to_parse)
        return INVALID_TIMESTAMP

    if ":" not in timestamp_to_parse:
        # add missing colons
        timestamp_to_parse = (
            timestamp_to_parse[:2] + ":" + timestamp_to_parse[2:4] + ":" + timestamp_to_parse[4:]
        )

    if timestamp_to_parse.count(":") == 1:
        chars_after_colon = timestamp_to_parse.split(":")[1]
        if len(chars_after_colon) == 3:
            # add zero
            chars_after_colon = chars_after_colon[:2] + ":" + chars_after_colon[2] + "0"
        elif len(chars_after_colon) == 2:
            # add missing seconds
            chars_after_colon = chars_after_colon + ":00"
        elif len(chars_after_colon) == 1:
            # add missing seconds and minutes
            chars_after_colon = "0:00" + chars_after_colon
        else:
            chars_after_colon = chars_after_colon[:2] + ":" + chars_after_colon[2:]

        timestamp_to_parse = timestamp_to_parse.split(":")[0] + ":" + chars_after_colon

    timestamp_to_parse = timestamp_to_parse[:8]
    try:
        timestamp = datetime.strptime(timestamp_to_parse, "%H:%M:%S")
        timestamp = timestamp.strftime("%H:%M:%S")
    except ValueError:
        print("Unable to parse timestamp:", timestamp_to_parse)
        return INVALID_TIMESTAMP

    return timestamp


def text_detect(img) -> str:
    """Detect text in an image using pytesseract. Returns the text as a string."""

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # inverse the image
    thresh = 255 - thresh

    # add padding
    with_border = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    text = pytesseract.image_to_string(with_border, config=r"--oem 3 --psm 6")

    clean_text = text.split("_")[-1].strip()
    clean_text = "".join(c for c in clean_text if (c.isdigit() or c == ":"))
    cleaned = clean_timestamp(clean_text)

    return cleaned
