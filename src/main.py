import argparse
from src.utils.motion_cap import motion_detector
from colorama import init
from termcolor import colored
from src.constants import PYTESSERACT_EXE
import pytesseract


def startup_message():

    init()

    print(
        colored(
            """
    Welcome to Bee Hotel Motion Capture! 🐝🏨    
                                                 
             __         .' '.                    
        _/__)        .   .       .               
       (8|)_}}- .      .        .                
        `\__)    '. . ' ' .  . '                 
                                                 \n""",
            "blue",
            "on_yellow",
        )
    )


def validate_args(args):
    if len(args.timestamp_rect) != 4:
        raise ValueError("Timestamp rectangle must be a list of 4 integers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", help="Path to video file")
    parser.add_argument(
        "--timestamp",
        "-t",
        help="Toggle if there is a timestamp in the video. Default is False",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-r",
        "--timestamp-rect",
        nargs="+",
        help="The timestamp rect (x1, y1, x2, y2). Default is (210, 20, 510, 50)",
        default=(210, 20, 510, 50),
    )
    parser.add_argument("--log", "-l", help="filepath to log file")
    parser.add_argument(
        "--hl",
        help="Run headless (don't show the video while processing)",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    validate_args(args)

    startup_message()

    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_EXE

    motion_detector(
        args.video,
        timestamp=args.timestamp,
        timestamp_rect=args.timestamp_rect,
        log=args.log,
        show=not args.hl,
        min_contour_area=400,
    )
