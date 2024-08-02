import streamlit as st


try:
    st.set_page_config(
        page_title="Execution Flow",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
finally:
    pass


def side_bar_view():
    st.header("Model Configurations")
    st.info("Check out the documentation "
            "at [link](https://docs.ultralytics.com/modes/predict/#inference-sources)")

    key = "sidebar"
    with st.form(f"{key}_upload", clear_on_submit=True):
        upload_image = st.file_uploader(
            "Upload Image(s)",
            accept_multiple_files=False,
            type=["png", "jpg", "jpeg"],
            key=f"{key}_upload_images"
        )

        col1, col2 = st.columns(2)
        with col1:
            augment = st.radio(
                "Augment",
                (True, False),
                horizontal=True
            )
        with col2:
            agnostic_nms = st.radio(
                "Agnostic NMS",
                (True, False),
                horizontal=True
            )
        image_size = st.number_input(
            "Image Size",
            value=640,
            step=32,
            min_value=640,
            max_value=1280
        )
        min_iou = st.slider(
            "Minimum IOU",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        min_confident_score = st.slider(
            "Minimum Confidence Score",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01
        )

        submit_btn = st.form_submit_button(
            label="Upload",
            type="primary",
            use_container_width=True
        )


def col_1_view():
    st.image("m10.jpg")


def col_2_view():
    dummy_counting_dct = {
        "Person": 1
    }

    with st.container(border=True):
        st.markdown("**Counting**")
        st.json(dummy_counting_dct)

    with st.expander(label="Object Detail", expanded=True):
        cls = st.selectbox(label="Class", options=["Person", "Animal"], index=0)

        st.markdown(f"Confident score :red[0.92]")


with st.sidebar:
    side_bar_view()

image_col, info_col = st.columns([8, 2])

with image_col:
    col_1_view()

with info_col:
    col_2_view()
