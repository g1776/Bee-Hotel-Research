import argparse
from src.motion_cap import motion_detector
from colorama import init
from termcolor import colored
from dotenv import load_dotenv
from src.config import MotionCapConfig


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
    # TODO: Move this to config.py
    if len(args["timestamp_rect"]) != 4:
        raise ValueError("Timestamp rectangle must be a list of 4 integers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to .env file. Default is .env",
        default=".env",
    )
    args = vars(parser.parse_args())

    # Load the .env file
    load_dotenv(args["config"])

    startup_message()

    motion_detector(config=MotionCapConfig())
