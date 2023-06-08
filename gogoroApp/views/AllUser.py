import streamlit as st
import pandas as pd
import plost

#尖／離峰騎乘：折線圖
#平／假日換電：長條圖
#尖／離峰換電：長條圖
#騎乘距離分群：長條圖

def load_view():
    # st.set_page_config(layout='wide', initial_sidebar_state='expanded')

    with open('styles/alluser.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
    st.sidebar.header('ALL USER INSIGHT')

    # st.sidebar.subheader('Heat map parameter')
    # time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 
    
    # st.sidebar.subheader('Donut chart parameter')
    # donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

    # st.sidebar.subheader('Line chart parameters')
    # plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
    # plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)
    
    
    #most_common_group 騎乘距離
    st.sidebar.subheader('騎乘距離四分位數')
    whichGroup = st.sidebar.multiselect('Which riding distance group ?', ('Group 1', 'Group 2','Group 3','Group 4')) 
    
    st.sidebar.subheader('Prefer Weekday or Weekend riding?')
    weekdayorWeekend = st.sidebar.multiselect('Weekday / Weekend', ('偏好平日', '偏好假日')) 

    st.sidebar.subheader('Prefer Peak or offPeak swapping?')
    PeakoroffPeak = st.sidebar.multiselect('Peak / offPeak', ('偏好尖峰', '偏好離峰')) 
    
    
    # Row A
    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")
    
    
    labeldatadf = pd.read_csv('dataset/user/label.csv',encoding='utf-8')
    outputdatadf = pd.read_csv('dataset/user/output.csv')
    
    
    st.markdown('### Peak or off Peak swapping Count Line chart')
    peak_offpeak_data = labeldatadf[['scooter_id', 'offPeak_count_x', 'peak_count_x', 'swapHour_preference_x']]
    # st.line_chart(labeldatadf, x = 'weekend_count_x', y = PeakoroffPeak)

    
    # Row B
    # seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
    # stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

    # c1, c2 = st.columns((7,3))
    # with c1:
    #     st.markdown('### Heatmap')
    #     plost.time_hist(
    #     data=seattle_weather,
    #     date='date',
    #     x_unit='week',
    #     y_unit='day',
    #     color=time_hist_color,
    #     aggregate='median',
    #     legend=None,
    #     height=345,
    #     use_container_width=True)
    # with c2:
    #     st.markdown('### Donut chart')
    #     plost.donut_chart(
    #         data=stocks,
    #         theta=donut_theta,
    #         color='company',
    #         legend='bottom', 
    #         use_container_width=True)

    # Row C
    # st.markdown('### Line chart')
    # st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)