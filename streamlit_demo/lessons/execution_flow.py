import datetime
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


def experiment_1():
    st.code('''
    st.info(str(datetime.datetime.now()))

    magic_number = st.slider("Magic number", min_value=0., max_value=1., step=0.1)
    print(magic_number)

    btn = st.button("Submit")
    input_a = None
    if btn:
        print("Enter btn function", datetime.datetime.now())
        st.toast("Button pressed")
        input_a = f"Hello word. Your magic number is: {magic_number}"

    st.info(magic_number)
    st.info(input_a)
    ''', language="python")

    st.info(str(datetime.datetime.now()))

    magic_number = st.slider("Magic number", min_value=0., max_value=1., step=0.1)
    print(magic_number)

    btn = st.button("Submit")
    input_a = None
    if btn:
        print("Enter btn function", datetime.datetime.now())
        st.toast("Button pressed")
        input_a = f"Hello word. Your magic number is: {magic_number}"

    st.info(magic_number)
    st.info(input_a)


def experiment_2():
    st.code('''
    st.info(str(datetime.datetime.now()))

    with st.form("form", clear_on_submit=True):
        magic_number = st.slider("Magic number", min_value=0., max_value=1., step=0.1)
        print(magic_number)

        btn = st.form_submit_button("Submit")

        if btn:
            print("Enter btn function", datetime.datetime.now())
            st.info("Hello World")
            
    st.info(magic_number)
    ''', language="python")
    st.info(str(datetime.datetime.now()))

    with st.form("form", clear_on_submit=True):
        magic_number = st.slider("Magic number", min_value=0., max_value=1., step=0.1)
        print(magic_number)

        btn = st.form_submit_button("Submit")

        if btn:
            print("Enter btn function", datetime.datetime.now())
            st.info("Hello World")

    st.info(magic_number)


def experiment_3():

    pass


cols = st.columns(2)

with cols[0]:
    experiment_1()


with cols[1]:
    experiment_2()

