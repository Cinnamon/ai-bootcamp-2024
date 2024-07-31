from abc import ABC, abstractmethod

import streamlit as st

import constants as c
from shared.crud.feedbacks import FeedbackCRUD
from shared.models.engine import Session
from shared.utils.files import save_uploaded_file
from shared.schemas import ModelInput, ModelOutput, Parameters, EditedOutput
from shared.models_ai import get_ai_model, BaseAIModel
from shared.views.canvas.canvas import st_annotate_tool


class BaseView(ABC):
    @abstractmethod
    def view(self, key: str):
        ...


class UploadView(BaseView):
    def view(self, key: str) -> ModelInput | None:
        with st.form(f"{key}_upload", clear_on_submit=True):
            upload_image = st.file_uploader(
                "Upload Image(s)",
                accept_multiple_files=False,
                type=c.YOLO_SUPPORTED_EXTENSIONS,
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

            if submit_btn:
                upload_image_path: str = save_uploaded_file(
                    upload_image,
                    c.USER_DATA_DIR
                )

                input_params = {
                    "augment": augment,
                    "agnostic_nms": agnostic_nms,
                    "image_size": image_size,
                    "min_iou": min_iou,
                    "min_confident_score": min_confident_score
                }

                return ModelInput(
                    upload_image=upload_image_path,
                    params=Parameters(**input_params)
                )

        return


class ImagePanelView(BaseView):
    def view(self, key: str, model_output: ModelOutput, image_path: str):
        updated_output, selected_index = st_annotate_tool(
            regions=model_output,
            background_image=image_path,
            key=f"{key}_visual",
            canvas_height=900,
            canvas_width=900
        )

        updated_output: ModelOutput
        selected_index: int

        return updated_output, selected_index


class InfoPanelView(BaseView):
    def view(self, key: str, model_output: ModelOutput, selected_index: int) -> EditedOutput | None:
        # Counting bboxes
        cls_name_dict: dict[str, int] = {c.CLASSES[k]: v for k, v in model_output.count().items()}

        with st.container(border=True):
            st.markdown("**Counting**")
            st.json(cls_name_dict)

        # View selected bbox
        if 0 <= selected_index < len(model_output.xyxysc):
            x_min, y_min, x_max, y_max, score, cls = model_output.xyxysc[selected_index]

            with st.expander(label="Object Detail", expanded=True):
                cls = st.selectbox(label="Class", options=c.CLASSES, index=int(cls))

                score_in_str = "%.3f" % score
                st.markdown(f"Confident score :red[{score_in_str}]")

                cls_index: int = c.CLASSES.index(cls)

            return EditedOutput(cls=cls_index)


class App(BaseView):
    def __init__(self):
        self._upload_view = UploadView()
        self._image_panel_view = ImagePanelView()
        self._info_panel_view = InfoPanelView()

        self._ai_model: BaseAIModel = get_ai_model(
            c.AI_MODEL,
            c.AI_MODEL_CONFIGS[c.AI_MODEL]
        )

        self.feedback_crud: FeedbackCRUD = FeedbackCRUD(
            session=Session()
        )

    @property
    def model_input(self) -> ModelInput | None:
        return st.session_state.get("model_input", None)

    @model_input.setter
    def model_input(self, model_in: ModelInput):
        st.session_state["model_input"] = model_in

    @property
    def model_output(self):
        return st.session_state.get("model_output", None)

    @model_output.setter
    def model_output(self, model_output: ModelOutput):
        st.session_state["model_output"] = model_output

    def view(self, key: str):
        with st.sidebar:
            st.header("Model Configurations")
            st.info("Check out the documentation "
                    "at [link](https://docs.ultralytics.com/modes/predict/#inference-sources)")

            model_input: ModelInput | None = self._upload_view.view(key=f"{key}_upload_inputs")
            if model_input is not None:
                # Run AI model when get new input
                with st.spinner("Running AI...."):
                    model_output: ModelOutput = self._ai_model.process(
                        image_in=model_input.upload_image,
                        params=model_input.params
                    )
                    st.toast("Finished AI processing", icon="🎉")
                self.model_input = model_input
                self.model_output = model_output

            if self.model_input is None:
                return

        image_col, info_col = st.columns([8, 2])
        with image_col:
            updated_model_output, selected_index = self._image_panel_view.view(
                key=f"{key}_images",
                model_output=self.model_output,
                image_path=self.model_input.upload_image
            )
            self.model_output = updated_model_output

        with info_col:
            edited_output: EditedOutput | None = self._info_panel_view.view(
                key=f"{key}_info",
                model_output=self.model_output,
                selected_index=selected_index
            )

            save = st.button(
                "Edit & Save",
                key=f"{key}_save_btn",
                use_container_width=True,
                type="primary"
            )

            if save and edited_output and 0 <= selected_index <= len(updated_model_output):
                updated_model_output[selected_index][-2] = edited_output.cls
                self.model_output = updated_model_output

                self.feedback_crud.create(
                    image_path=self.model_input.upload_image,
                    data=self.model_output.to_dict()
                )

                st.toast("Saved", icon="🎉")
