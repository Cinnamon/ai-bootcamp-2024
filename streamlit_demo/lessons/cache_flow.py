import json
import time

import streamlit as st
import pandas as pd

from functools import lru_cache


class Model:
    def __init__(self, dct):
        self.dct = dct


def experiment_1():
    st.code('''
    class Model:
    def __init__(self, dct):
        self.dct = dct
    
    # @lru_cache(1)
    # @st.cache_data
    @st.cache_resource
    def load_data(dct: dict) -> Model:
        print("I will go sleep for 3s")
        time.sleep(3)

        model = Model(dct)
        return model

    data_dct = {
        'Column1': [1, 2, 3, 4, 5],
        'Column2': ['A', 'B', 'C', 'D', 'E'],
        'Column3': [10.5, 20.5, 30.5, 40.5, 50.5],
        'Column4': [True, False, True, False, True]
    }

    model = load_data(data_dct)
    st.json(model.dct)

    model.dct = {}
    st.button("Rerun")
    ''', language='python')

    # @lru_cache(1)
    @st.cache_data
    # @st.cache_resource
    def load_data(dct: dict) -> Model:
        print("I will go sleep for 3s")
        time.sleep(3)

        model = Model(dct)
        return model

    data_dct = {
        'Column1': [1, 2, 3, 4, 5],
        'Column2': ['A', 'B', 'C', 'D', 'E'],
        'Column3': [10.5, 20.5, 30.5, 40.5, 50.5],
        'Column4': [True, False, True, False, True]
    }

    model = load_data(data_dct)
    st.json(model.dct)

    btn = st.button("Rerun")
    if btn:
        model.dct = {}


experiment_1()
