import streamlit as st
import pandas as pd

from config import road_config, predict_config
from tools.info import get_result
from tools.display import show_map, actual_predict_val, graph_result

import datetime

st.set_page_config(layout='wide')

if not "initialized" in st.session_state:
    st.session_state.road = 'Highway No. 1'
    # st.session_state.date = datetime.date(2020, 1, 2)
    st.session_state.time = '00:00'
    st.session_state.result = None
    st.session_state.old_road = 0
    st.session_state.initialized = True

st.title(f"🚚 Travel Time Prediction")
with st.expander(f"ℹ️ - About this research"):
    st.write(
        """
        - Member in group is ***Rathachai Chawuthai, Nachaphat Ainthong, Surasee Intarawart and Niracha Boonyanaet.***
        """
    )
    st.write("")
    # st.markdown(f"- Member in group is ***Rathachai Chawuthai, Nachaphat Ainthong, Surasee Intarawart and Niracha Boonyanaet.***")

column1, column2 = st.columns((1, 1))
with column2:
    st.subheader('📝 Insert information for predict travel time')
    with st.form('filter'):
        st.selectbox(
            '🚩 Select the Highway Nunber.',
            ['Highway No. 1', 'Highway No. 2', 'Highway No. 4', 'Highway No. 7', 'Highway No. 9', 'Highway No. 32', 'Highway No. 35', 'Highway No. 41', 'Highway No. 304', 'Highway No. 331'],
            key='road'
        )

        st.date_input(
            "🌛 Select date.",
            min_value=datetime.date(2020, 1, 2),
            max_value=datetime.date(2020, 4, 30),
            value=datetime.date(2020, 1, 2),
            key='date'
        )

        st.selectbox(
            '⏰ Select time.',
            ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00'],
            key='time'
        )
        submit = st.form_submit_button('Process')

    road_num = int(st.session_state.road.split(' ')[-1])
    if st.session_state.old_road != road_num:
        st.session_state.result = get_result(predict_config.DATA_PATH, predict_config.MODEL_WEIGHT_PATH, road_num)
        st.session_state.old_road = road_num
    actual_predict_val(st.session_state.result, st.session_state.date, st.session_state.time)
    
with column1:
    road_num = int(st.session_state.road.split(' ')[-1])
    show_map(road_config.ROAD_DATA_PATH, road_num)

graph_result(st.session_state.result, int(st.session_state.road.split(' ')[-1]))


    

