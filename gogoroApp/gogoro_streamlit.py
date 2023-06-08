import streamlit as st
import utils as utl
from views import PerUser,Risk,AllUser

st.set_page_config(layout="wide", page_title='Gogoro Living Circle',page_icon="🛵")
# st.image('output/driver.jpg')

st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()
def navigation():
    route = utl.get_current_route()
    if route == "PerUser":
        PerUser.load_view()
    elif route == "Risk":
        Risk.load_view()
    elif route == "AllUser":
        AllUser.load_view()
    elif route == None:
        PerUser.load_view()
if __name__ == "__main__":
    
    # set the config
    navigation()
    