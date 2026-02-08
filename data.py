# HOW TO RUN THIS APP:
# 1. Save this file as `data.py`.
# 2. Place your dataset (e.g., normalized_bankruptcy_dataset.csv) in the same folder.
# 3. Run the main app.py file with:
#    streamlit run app.py

import streamlit as st
import pandas as pd

def set_beautiful_background():
    """Custom CSS for gradient + glassmorphism theme."""
    page_styles = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main .block-container {
        background-color: rgba(20, 20, 40, 0.7);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    h1, h2, p, .stDataFrame, label {
        color: #FFFFFF !important;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* This targets the main content area specifically for this page */
    .main > div:first-child > .block-container {
       background-color: rgba(20, 20, 40, 0.7);
    }

    button[title='Download as CSV'] {
        display: none;
    }
    </style>
    """
    st.markdown(page_styles, unsafe_allow_html=True)

def add_app_footer():
    """Adds a custom footer to the page."""
    footer_styles = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(15, 12, 41, 0.8); /* Matches the dark background */
        color: #FFFFFF;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.18);
    }
    </style>
    """
    footer_html = """
    <div class="footer">
        <p>@2025 </p>
    </div>
    """
    st.markdown(footer_styles, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    """Load only the first 2000 rows from a local CSV file."""
    try:
        df = pd.read_csv("normalized_bankruptcy_dataset.csv", nrows=2000)
        return df
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please check the filename and location.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

def app():
    """Main function for the Data Viewer page."""
    # st.set_page_config was removed from here.
    set_beautiful_background()

    st.title("🏦 Bankruptcy Dataset Viewer")
    st.markdown("Displays a preview of the bankruptcy dataset.")

    df = load_dataset()

    if df is not None:
        st.header("📊 Dataset Preview ")
        table_height = 450
        st.dataframe(df, height=table_height)
    else:
        st.warning("Please ensure the CSV file exists in the project folder.")

    add_app_footer()

# The if __name__ == "__main__": block is removed as this file is a module.
