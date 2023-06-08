import streamlit as st
from streamlit.components.v1 import html
import streamlit.components.v1 as components

from PATHS import NAVBAR_PATHS


def inject_custom_css():
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def get_current_route():
    try:
        return st.experimental_get_query_params()['nav'][0]
    except:
        return None


def navbar_component():


    navbar_items = ''
    for key, value in NAVBAR_PATHS.items():
        navbar_items += (f'<a  class="navitem" href="/?nav={value}">{key}</a>')

   
    component = rf'''
            <nav class="container navbar" id="navbar">
                <ul class="navlist">
                {navbar_items}
                    <a class="navitem1" href="https://www.gogoro.com/">GOGORO LIVING CIRCLE ENGINE</a>
                </ul>
                
            </nav>
            '''


    st.markdown(component, unsafe_allow_html=True)
    js = '''
    <script>
        // navbar elements
        var navigationTabs = window.parent.document.getElementsByClassName("navitem");
        var cleanNavbar = function(navigation_element) {
            navigation_element.removeAttribute('target')
        }
        
        for (var i = 0; i < navigationTabs.length; i++) {
            cleanNavbar(navigationTabs[i]);
        }
        
        
    </script>
    '''
    html(js)