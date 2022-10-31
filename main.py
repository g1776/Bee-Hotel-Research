import argparse
from motion_cap import motion_detector
from colorama import init
from termcolor import colored
from constants import PYTESSERACT_EXE
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", help="Path to video file")
    parser.add_argument(
        "--timestamp",
        "-t",
        help="Toggle if there is a timestamp in the video. This will black out the timestamp",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--log", "-l", help="filepath to log file")
    parser.add_argument(
        "--hl",
        help="Run headless (don't show the video while processing)",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    startup_message()

    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_EXE

    motion_detector(
        args.video,
        timestamp=args.timestamp,
        timestamp_rect=(210, 20, 510, 50),
        log=args.log,
        show=not args.hl,
        min_contour_area=400,
    )
