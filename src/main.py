import argparse
from src.motion_cap import motion_detector
from colorama import init
from termcolor import colored
import pytesseract
from dotenv import load_dotenv
from types import SimpleNamespace
from src.config import MotionCapConfig
import os


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
    parser.add_argument(
        "--tesseract",
        help="Path to tesseract executable. Default is 'tesseract'",
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to .env file. Default is .env",
        default=".env",
    )
    args = parser.parse_args()

    # either use the commandline args or the .env file
    if args.config:
        load_dotenv(args.config)
        args = SimpleNamespace(
            **{k: os.getenv(k.upper()) or vars(args)[k] for k in vars(args).keys()}
        )
    else:
        validate_args(args)

    startup_message()

    pytesseract.pytesseract.tesseract_cmd = args.tesseract

    motion_detector(config=MotionCapConfig(args))
