from turtle import width
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime

from sklearn.metrics import mean_absolute_error

road_centroid = {
    1: (15.561914417289183, 16.85901078855188),
    2: (14.850897312327184, 16.27010669391355),
    4: (11.495837799354899, 13.314148102062967),
    7: (13.113253464736975, 13.73277648540468),
    9: (13.642726998935187, 13.649629122634497),
    32: (14.198731291192727, 15.275452778629077),
    41: (8.167021308334858, 9.946246687710023),
    304: (13.879089398416676, 14.388486307650279),
    35: (13.34104092128205, 13.640325795495965),
    331: (13.11827478284012, 13.59731304067158), 
}

road_kg = {
    1: 322,
    2: 232,
    4: 219,
    7: 95,
    9: 149,
    32: 143,
    41: 242,
    304: 104,
    35: 75,
    331: 80,
}

def show_map(road_data_dir, road_num):
    if road_num == 331:
        df_road = pd.read_csv(road_data_dir + '/roaddb.csv')
        df_road = df_road[(df_road['rd'] == road_num) & (df_road['lat'] >= 13.09200) & (df_road['lat'] <= road_centroid[road_num][1])].reset_index(drop=True)
        df_road['Road Segments'] = df_road.apply(lambda row: f"Highway No. {(row['rd'])}", axis=1)
    elif road_num == 7:
        df_road = pd.read_csv(road_data_dir + '/roaddb.csv')
        df_road = df_road[(df_road['rd'] == road_num) & (df_road['lat'] >= road_centroid[road_num][0]) & (df_road['lat'] <= road_centroid[road_num][1]) & (df_road['lon'] > 100.7886)].reset_index(drop=True)
        df_road = df_road[(df_road['lat'] < 13.56424) | (df_road['lat'] > 13.5985) | (df_road['lon'] < 100.9485) | (df_road['lon'] > 100.9539)].reset_index(drop=True)
        df_road = df_road[((df_road['lat'] < 13.2990) | (df_road['lat'] > 13.33408)) | ((df_road['lon'] < 100.989) | (df_road['lon'] > 100.9945))].reset_index(drop=True)
        df_road['Road Segments'] = df_road.apply(lambda row: f"Highway No. {(row['rd'])}", axis=1)
    elif road_num == 304:
        df_road = pd.read_csv(road_data_dir + '/roaddb.csv')
        df_road = df_road[(df_road['rd'] == road_num) & (df_road['lat'] >= road_centroid[road_num][0]) & (df_road['lat'] <= road_centroid[road_num][1]) & (df_road['lon'] > 100.5904)].reset_index(drop=True)
        df_road['Road Segments'] = df_road.apply(lambda row: f"Highway No. {(row['rd'])}", axis=1)
    elif road_num == 9:
        df_road = pd.read_csv(road_data_dir + '/roaddb.csv')
        df_road = df_road[(df_road['rd'] == road_num)].reset_index(drop=True)
        df_road = df_road[((df_road['lat'] > road_centroid[road_num][0]) | (df_road['lat'] > road_centroid[road_num][1])) | ((df_road['lon'] < 100.41451712861084) | (df_road['lon'] > 100.68650542608228))].reset_index(drop=True)
        df_road['Road Segments'] = df_road.apply(lambda row: f"Highway No. {(row['rd'])}", axis=1)
        # df_road
    else:
        df_road = pd.read_csv(road_data_dir + '/roaddb.csv')
        df_road = df_road[(df_road['rd'] == road_num) & (df_road['lat'] >= road_centroid[road_num][0]) & (df_road['lat'] <= road_centroid[road_num][1])].reset_index(drop=True)
        df_road['Road Segments'] = df_road.apply(lambda row: f"Highway No. {(row['rd'])}", axis=1)

    if road_num in [7, 9, 32, 35, 304, 331]:
        zoom = 8
    else:
        zoom = 7
    fig = px.scatter_mapbox(df_road, lat='lat', lon='lon', width=800, height=800, zoom=zoom)
    fig.update_layout(autosize=False, width=800, height=800, margin=dict(l=0,r=0,b=0,t=0,pad=0), mapbox_style='open-street-map')
    st.plotly_chart(fig, use_container_width=True)


def actual_predict_val(result, date, time):
    time = int(time.split(':')[0])
    time_select = datetime.datetime(date.year, date.month, date.day, time)

    out = result[result['time'] == time_select]

    target = int(out[out['type'] == 'target']['delta_t'].values[0] // 60)
    pred = int(out[out['type'] == 'predict']['delta_t'].values[0] // 60)
    # delta = target - pred
    st.metric('Actual Travel Time', f"{target} min")
    st.metric('Predict Travel Time', f"{pred} min")
    # st.metric("Delta time between Target and Predict", f"{delta} Min")


def graph_result(result, road):
    target = result[result['type'] == 'target']['delta_t']
    pred = result[result['type'] == 'predict']['delta_t']
    avg_target = target.mean()
    mae = mean_absolute_error(target, pred)

    avg_target = int(avg_target // 60)
    mae = int(mae // 60)

    with st.expander(f"📌 Summary of travel time data on Highway No.{road}"):
        cols = st.columns(4)
        cols[0].metric('Road segment', f"Highway No.{road}")
        cols[1].metric('Distance of road segment', f"{road_kg[road]} km")
        cols[2].metric('Average travel time', f"{avg_target} min")
        cols[3].metric('Mean Absolute Error', f"{mae} min")
       

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=result[result['type'] == 'target']['time'], 
                y=result[result['type'] == 'target']['delta_t'] / 60, 
                name='Actual',
                line=dict(color='#001F3F')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=result[result['type'] == 'predict']['time'], 
                y=result[result['type'] == 'predict']['delta_t'] / 60, 
                name='Predict',
                line=dict(color='#FF4136')
            )
        )
        fig.update_layout(
            title=f"Actual and predict travel time in Jan 2020 - Apr 2020 on Highway No.{road}",
            xaxis_title='Time',
            yaxis_title='Travel time (min)',
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)