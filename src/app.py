import streamlit as st

from src.motion_cap import motion_detector
from src.config import MotionCapConfig
from dotenv import load_dotenv
import sys
from streamlit import runtime
from streamlit.web import cli

# Streamlit app code
def main():

    st.markdown(
        """
    # Bee Hotel Motion Detection and Identification
    """
    )

    # load default config
    DEFAULT_CONFIG = ".env"
    load_dotenv(DEFAULT_CONFIG)
    config = MotionCapConfig()

    config_fp = st.sidebar.text_input("Config file", ".env")
    st.sidebar.markdown(
        "<p><i>Leave this field empty and hit update to manually enter the config values</i></p>",
        unsafe_allow_html=True,
    )

    update_config = st.sidebar.button("Update config")

    config_container = st.sidebar.container()

    if update_config:
        load_dotenv(config_fp)
        config = MotionCapConfig()

        # clear config container
        config_container.empty()

    # populate config container
    for key, value in config.__dict__.items():
        config_container.markdown(f"<b>{key}</b>: {value}", unsafe_allow_html=True)

    start = st.button("Start")

    header = st.empty()

    # create side-by-side containers
    col1, col2 = st.columns(2)
    placeholder = col1.empty()

    def imshow_callback(image):
        placeholder.image(image)

    def logging_callback(log):
        col2.markdown(log)

    if start:
        motion_detector(config, imshow_callback=imshow_callback, logging_callback=logging_callback)
        header.markdown("## Motion Detection Feed")


# Driver Code
if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(cli.main())
