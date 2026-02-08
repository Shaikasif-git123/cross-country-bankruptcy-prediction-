import streamlit as st
from streamlit_option_menu import option_menu

# Import the dedicated page modules
# Make sure these python files are in the same directory, each with an app() function.

import home
import aaa
import pre
import data
import pre1

# Set the page configuration for a nice, clean look
st.set_page_config(
    page_title="Bankruptcy",
    layout="wide"
)

# This is the main function that runs the entire app
def main():
    """Main function to run the Streamlit app."""
    with st.sidebar:
        selected = option_menu(
            menu_title='Main Menu',
            # Corrected the page titles for better readability
            options=['Home', 'Visualization', 'Slide Prediction', 'Data', 'Input Prediction'],
            # Updated icons to better match the page names
            icons=['house-fill', 'pie-chart-fill', 'sliders', 'table', 'pencil-square'],
            menu_icon='cast',
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": '#0e1117'},
                "icon": {"color": "orange", "font-size": "23px"},
                "nav-link": {"color": "white", "font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#262730"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

    # Use an if/elif block to route to the correct page based on the user's selection
    if selected == "Home":
        home.app()
    elif selected == "Visualization":
        aaa.app()
    elif selected == "Slide Prediction":
        pre.app()
    elif selected == "Data":
        data.app()
    elif selected == "Input Prediction":
        pre1.app()

if __name__ == "__main__":
    main()

