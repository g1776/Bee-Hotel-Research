import streamlit as st

st.set_page_config(
    page_title="Bee Hotel Motion Detection and Identification",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.motion_cap import motion_detector
from src.eval.utils import parse_log_entry
from src.config import MotionCapConfig
from dotenv import load_dotenv
import sys
import plotly.express as px
from streamlit import runtime
from streamlit.web import cli
import pandas as pd
import os

# Streamlit app code
def main():

    st.markdown(
        """
    ## Bee Hotel Motion Detection and Identification
    """
    )

    def load_config():

        # load default config
        DEFAULT_CONFIG = ".env"
        load_dotenv(DEFAULT_CONFIG)
        config = MotionCapConfig()

        for key, _ in config.__dict__.items():
            del os.environ[key]

        st.sidebar.markdown("# Config")
        st.sidebar.markdown(
            f"To change the config, edit the `{DEFAULT_CONFIG}` file and refresh the page."
        )

        config_container = st.sidebar.container()

        # populate config container
        for key, value in config.__dict__.items():
            config_container.markdown(f"<b>{key}</b>: {value}", unsafe_allow_html=True)

        return config

    config = load_config()

    start = st.button("Start")

    header = st.empty()

    # create side-by-side containers
    col1, col2 = st.columns([2, 1])

    placeholder_img = col1.empty()
    placeholder_log = col2.empty()
    plotly_placeholder = st.empty()

    def imshow_callback(image):
        placeholder_img.image(image, use_column_width="always")

    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    def logging_callback(log):
        # update log
        parsed = parse_log_entry(log, verbose=False)
        if parsed is not None:
            # drop "timestamp was edited" key
            parsed = {k: v for k, v in parsed.items() if k != "timestamp was edited"}

            st.session_state["logs"] = [parsed] + st.session_state["logs"]

        placeholder_log.dataframe(st.session_state["logs"])

        # update plot
        if len(st.session_state["logs"]) > 0:
            data = pd.DataFrame(st.session_state["logs"])
            fig = px.scatter(x=data["timestamp"], y=data["bee_id"])
            fig.update_layout(
                xaxis_title="Timestamp",
                yaxis_title="Bee ID",
                title="Bee IDs Identified over Time",
                title_font_size=25,
                hoverlabel=dict(
                    font_size=20,
                ),
                yaxis=dict(tickfont=dict(size=20)),
                xaxis=dict(tickfont=dict(size=20)),
            )
            # increase hover text size
            plotly_placeholder.plotly_chart(fig, use_container_width=True)

    if start:
        motion_detector(config, imshow_callback=imshow_callback, logging_callback=logging_callback)
        header.markdown("## Motion Detection Feed")

        plotly_placeholder = st.empty()


# Driver Code
if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(cli.main())
