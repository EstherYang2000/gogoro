import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import streamlit as st
from streamlit_folium import st_folium, folium_static
import leafmap.foliumap as leafmap
import ssl

APP_TITLE = 'Scooter Accident in Taiwan'
APP_SUB_TITLE = 'Gogoro Network X NTU DAC'
ssl._create_default_https_context = ssl._create_unverified_context
def zoom_slider():
    default_zoom = 10
    zoom = st.sidebar.slider(
        "調整地圖大小", min_value=5, max_value=20, value=default_zoom)
    return zoom


def radius_slider():
    default_radius = 15
    radius = st.sidebar.slider(
        "調整半徑", min_value=1, max_value=40, value=default_radius)
    return radius


def types_selectbox():
    types = st.sidebar.selectbox(
        "事故類型", ["A1+A2 所有事故", "A1 死亡事故", "A2 受傷事故"], 2)

    return types


def hour_slider():
    default_hour = 1
    hour = st.sidebar.slider("選擇當日時間", min_value=0,
                             max_value=23, value=default_hour)
    return hour


def display_map(data, radius, hour, zoom):
    data = data[data["hour"] == hour]

    st.title("Heatmap")

    m = leafmap.Map(center=[22.541141, 120.372052],
                    zoom=zoom, zoom_control=False,
                    scrollWheelZoom=False,)
    m.add_heatmap(
        data,
        latitude="lat",
        longitude="lng",
        value="danger_value",
        name="Heat map",
        radius=radius,
        max_opacity=0.8,
    )
    m.to_streamlit(height=700)


def main():
    #st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    # Load Data
    df_A1 = pd.read_parquet("dataset/risk/A1_heatmapData.parquet")
    df_A2 = pd.read_parquet("dataset/risk/A1_heatmapData.parquet")
    df_A1A2 = pd.concat([df_A1, df_A2], axis=0)

    types = types_selectbox()
    zoom = zoom_slider()
    radius = radius_slider()
    hour = hour_slider()

    if types == "A1+A2 所有事故":
        data = df_A1A2
    elif types == "A1 死亡事故":
        data = df_A1
    elif types == "A2 受傷事故":
        data = df_A2

    display_map(data, radius, hour, zoom)

def load_view():
    main()