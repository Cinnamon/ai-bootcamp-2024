import streamlit as st


def set_page_config():
    try:
        st.set_page_config(
            page_title="Object Detection",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    finally:
        pass
