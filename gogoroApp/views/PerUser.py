import time
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap
from stqdm import stqdm
from streamlit_folium import folium_static
from folium.plugins import HeatMapWithTime


@st.cache_data(show_spinner='Fetching data...')
def importData(scooter_id):
    print("read data")
    ProbabilityFeather = f"output/{scooter_id}.feather"
    # read gaussian heatmap
    gaussian_df = pd.read_feather(ProbabilityFeather)
    heatmap_data = gaussian_df[['Latitude', 'Longitude',
                         'Probability']].values.tolist()
    SWAPOUT = f"output/{scooter_id}_swapout.shp"
    out_gdf = gpd.read_file(
        SWAPOUT, encoding="utf-8")
    SWAPIN = f"output/{scooter_id}_swapin.shp"
    in_gdf = gpd.read_file(
        SWAPIN, encoding="utf-8")
    RISK = f"risk/swapin.shp"
    risk_gdf = gpd.read_file(
        RISK, encoding="utf-8")
    # import 事故點 資料
    A1 = f"dataset/risk/A1_heatmapData.parquet"
    df_A1 = pd.read_parquet(A1)
    A2 =  f"dataset/risk/A2_heatmapData.parquet"
    df_A2 = pd.read_parquet(A2)
    maxlat = gaussian_df['Latitude'].max()
    minlat = gaussian_df['Latitude'].min()
    maxlng = gaussian_df['Longitude'].max()
    minlng = gaussian_df['Longitude'].min()
    df_A1 = df_A1[(df_A1['lat']>=minlat ) & (df_A1['lat']<=maxlat) & (df_A1['lng']>=minlng ) & (df_A1['lng']<=maxlng) ].reset_index(inplace=False)
    df_A2 = df_A2[(df_A2['lat']>=minlat ) & (df_A2['lat']<=maxlat) & (df_A2['lng']>=minlng ) & (df_A2['lng']<=maxlng) ].reset_index(inplace=False)
    df_A1A2 = pd.concat([df_A1, df_A2], axis=0)

    return heatmap_data, out_gdf, in_gdf, risk_gdf, gaussian_df,df_A1,df_A2,df_A1A2


def instyle_function(x): return {
    # "fillColor": "#576CBC",
    "color": "#F3F3F3",
    "fillOpacity": 0.00001,
    "weight": 0.5,
    "light_weight": 3
}


def outstyle_function(x): return {
    # "fillColor": "#FFD4D4",
    "color": "#90A5B2",
    "fillOpacity": 0.00001,
    "weight": 0.5,
}


def riskstyle_function(x): return {
    # "fillColor": "#B7B78A",
    "color": "#0069D2",
    "fillOpacity": 0.00001,
    "weight": 6,
}


def showMap(options, map_test, heatmap_data, out_gdf, in_gdf, risk_gdf,data, radius, zoom):

    if 'Heat Map' in options:
        # Add Heat Map layer to the map
        gradient = {
            0.1: 'blue',                   # Lower than Q1
            0.3: 'cyan',                  # Q1 and below
            0.5: 'yellow',                # Between Q1 and Q2
            0.7: 'red',                   # Between Q2 and Q3
            1: 'purple'  # Q3 and above
        }
        map_test.add_child(HeatMap(heatmap_data, gradient=gradient,
                                   min_opacity=0.000001,
                                   max_opacity=0.9,
                                   radius=6))  # .add_to(map_test)

    if 'Swap in Circle' in options:
        # Add Swap in Circle layer to the map
        map_test.add_child(folium.GeoJson(data=in_gdf["geometry"],
                                          zoom_on_click=True,
                                          style_function=instyle_function
                                          ))  # .add_to(map_test)

    if 'Swap out Circle' in options:
        # Add Swap out Circle layer to the map
        map_test.add_child(folium.GeoJson(data=out_gdf["geometry"],
                                          zoom_on_click=True,
                                          style_function=outstyle_function
                                          ))  # .add_to(map_test)

    if 'Accidents Points' in options:
        # Add Accident Points layer to the map
        map_test.add_child(folium.GeoJson(data=risk_gdf["geometry"],
                                          zoom_on_click=True,
                                          style_function=riskstyle_function,
    
                                          )) 
    # add risk layer
    if data is not None:
        heat_data = []
        for _, row in data.iterrows():
            heat_data.append([row['lat'], row['lng'], row['danger_value']])
        print(heat_data[0])
        map_test.add_child(HeatMap(heat_data, gradient={0.1: 'blue', 0.5: 'lime', 1: 'red'},
                                    zoom=zoom, 
                                    zoom_control=False,
                                    min_opacity=0.000001,
                                    max_opacity=0.9,
                                    radius=radius))  
            
    # Calculate bounds to zoom the map
    bounds = map_test.get_bounds()
    if bounds:
        map_test.fit_bounds(bounds)

    return map_test


def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


def danger_value(value):
    color = ''
    if value < 20:
        color = 'green'
    elif value < 35:
        color = 'yellow'
    elif value < 50:
        color = 'orange'
    else:
        color = 'red'

    return color

def zoom_slider():
    default_zoom = 10
    zoom = st.sidebar.slider(
        "調整地圖大小", min_value=6, max_value=14, value=default_zoom)
    return zoom


def radius_slider(zoom):
    radius = zoom + 4

    return radius



def load_view():
     
    initial_map = folium.Map(location=[23.6978, 120.9605],
                        tiles="cartodbpositron", zoom_start=8)
    # Display the initial map
    map_placeholder = st.empty()

    print("--------begin loading successful----------")
    #"03a6729f-8045-4ada-9110-3ec91079c83c"
    scooterid = st.sidebar.text_input("Enter a scooter id", key='user')
    options = st.sidebar.multiselect(
        'Whch layers do you wanna show on map ? ',
        ['Heat Map', 'Swap in Circle', 'Swap out Circle', 'Accidents Points'])
    # st.sidebar.image('output/driver.jpg')
    print(scooterid)
    print(options)

    riskoptions = st.sidebar.selectbox("Whch risk layers do you wanna show on map ?", options=[None,"A1+A2 所有事故", "A1 死亡事故", "A2 受傷事故"])
    print(riskoptions)
    zoom = zoom_slider()
    radius = radius_slider(zoom)

    
    if scooterid:
        heatmap_data, out_gdf, in_gdf, risk_gdf, gaussian_df,df_A1,df_A2,df_A1A2 = importData(scooterid)
        print("--------scooter id successful----------")
    else:
        heatmap_data, out_gdf, in_gdf, risk_gdf, gaussian_df, df_A1, df_A2, df_A1A2 = None, None, None, None, None, None, None, None
 
    
    if len(options) != 0  or (riskoptions is not None):

        # map_placeholder.empty()
        with map_placeholder.container():
            
            print("--------options successful----------")

            if riskoptions == "A1+A2 所有事故":
                data = df_A1A2
            elif riskoptions == "A1 死亡事故":
                data = df_A1
            elif riskoptions == "A2 受傷事故":
                data = df_A2  
            else:
                data = None
            
            initial_map = showMap(options, initial_map, heatmap_data, out_gdf, in_gdf, risk_gdf,data, radius, zoom)
            
            folium_static(initial_map)
            pro = st.progress(100, "Data loading")
            sc = st.success("Map generating succeed!", icon="✅")
        
            time.sleep(2)  # Wait for 3 seconds
            sc.empty()
            pro.empty()
       
    
    else:
        print("show the initial map")
        folium_static(initial_map)
    
    # Row A
    
    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")
    
    

    form = st.form(key="form_settings")
    col1, col2 = form.columns([3, 2])
    col1.radio("User's feature", ("flexible", "monotonic"))
    value = col2.markdown(
        f'''
        <h4 >Dangerous Value</h4>
        <h2 style="color: green;">0</h2>
        ''', unsafe_allow_html=True
    )
    form.form_submit_button(label="Refresh Program", on_click=clear_cache)
    
            
            
