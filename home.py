import streamlit as st
import base64
import os  # Recommended for handling file paths

def app():
    """This function runs the home page of the Streamlit app."""

    # --- Asset Management ---
    def get_image_as_base64(file_path):
        """Reads an image file and returns its base64 encoded string."""
        if not os.path.exists(file_path):
            st.error(f"Error: The image file was not found at {file_path}")
            return None
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # --- CSS Styling ---
    def add_custom_css(background_image_base64):
        """Injects custom CSS for background and footer into the app."""
        css_style = f"""
        <style>
        /* Main background settings */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{background_image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Container for centering title content */
        .centered-content {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 90vh; /* Adjust height to prevent overlap with footer */
            text-align: center;
            color: white;
        }}

        /* Style for the main title */
        .main-title {{
            font-size: 5rem;
            font-weight: bold;
            text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.7);
        }}

        /* Custom Footer Style */
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent black */
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 1rem;
        }}

        /* Hide the default Streamlit menu and footer, but NOT the header */
        #MainMenu, footer {{
            visibility: hidden;
        }}
        
        /* Make the header transparent to show the background */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        </style>
        """
        st.markdown(css_style, unsafe_allow_html=True)

    # --- Main Page Rendering ---
    # Make sure "background.jpg" is the correct name and it's in the same folder as your scripts.
    image_file = "background.jpg   .png" # Corrected filename
    img_base64 = get_image_as_base64(image_file)

    if img_base64:
        # Inject the CSS with the background image
        add_custom_css(img_base64)

        # Display the main centered content
        st.markdown("""
        <div class="centered-content">
            <div class="main-title">
                Advancing Corporate Bankruptcy Forecasting: A Multi-Model Approach with Enhanced Imbalance Handling
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display the custom footer
        st.markdown("""
        <div class="footer">
            @2025
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display a fallback message if the image can't be loaded
        st.title("Advancing Corporate Bankruptcy Forecasting")
        st.warning("Background image not found. Please ensure it is in the correct path.")

