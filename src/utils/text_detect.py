import cv2
from pytesseract import pytesseract


def text_detect(img) -> str:
    """Detect text in an image using pytesseract. Returns the text as a string."""

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # inverse the image
    thresh = 255 - thresh

    # add padding
    thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    text = pytesseract.image_to_string(thresh, config=r"--oem 3 --psm 6")

    clean_text = text.split("_")[-1].strip()
    clean_text = "".join(c for c in clean_text if (c.isdigit() or c == ":"))
    return clean_text
