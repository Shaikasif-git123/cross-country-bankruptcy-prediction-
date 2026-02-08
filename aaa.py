# HOW TO RUN THIS APP:
# 1. Save this code as a Python file (e.g., `visualizer.py`).
# 2. Make sure you have the `normalized_bankruptcy_dataset.csv` file in the EXACT same folder.
# 3. Open your terminal or command prompt.
# 4. Install the necessary libraries by running:
#    pip install streamlit pandas plotly
# 5. Navigate to the folder where you saved the file and the CSV.
# 6. Run the following command in your terminal:
#    streamlit run visualizer.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# This function loads the dataset and caches it for performance.
@st.cache_data
def load_data():
    """Loads the dataset from the CSV file."""
    try:
        # Assuming the file is named 'normalized_bankruptcy_dataset.csv'
        # and is in the same directory as the script.
        df = pd.read_csv("normalized_bankruptcy_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset 'normalized_bankruptcy_dataset.csv' not found. Please ensure it's in the same directory.")
        return None

def create_histogram(df, column):
    """
    Generates a histogram with distinct colors for each bankruptcy class.
    This helps in visualizing the distribution of a single variable.
    """
    fig = px.histogram(df, x=column, color='Bankrupt',
                       title=f'Distribution of {column} by Bankruptcy Status',
                       labels={column: column, 'Bankrupt': 'Bankrupt (1=Yes)'})
    return fig

def create_box_plot(df, column):
    """
    Generates a box plot to compare the distribution of a feature
    across bankrupt and non-bankrupt companies.
    """
    fig = px.box(df, x='Bankrupt', y=column,
               title=f'Box Plot of {column} vs. Bankruptcy Status',
               labels={'Bankrupt': 'Bankrupt (1=Yes)', column: column})
    return fig

def create_scatter_plot(df, x_column, y_column):
    """
    Generates a scatter plot to show the relationship between two features.
    The points are colored by their bankruptcy status.
    """
    fig = px.scatter(df, x=x_column, y=y_column, color='Bankrupt',
                     title=f'Scatter Plot of {x_column} vs. {y_column}',
                     labels={x_column: x_column, y_column: y_column, 'Bankrupt': 'Bankrupt (1=Yes)'})
    return fig

def create_correlation_heatmap(df):
    """
    Generates a correlation heatmap of all numerical features.
    This helps to identify which features are most strongly correlated.
    """
    # Select only the numerical columns for the correlation matrix
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numerical_df.corr()

    # Create the heatmap using Plotly Figure Factory
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Viridis',
        annotation_text=corr.round(2).values,
        showscale=True
    )
    fig.update_layout(title_text='<b>Correlation Heatmap</b>',
                      xaxis_showgrid=False,
                      yaxis_showgrid=False)
    return fig

def create_pie_chart(df):
    """
    Generates a pie chart to show the proportion of bankrupt vs. non-bankrupt companies.
    """
    bankrupt_counts = df['Bankrupt'].value_counts().reset_index()
    bankrupt_counts.columns = ['Bankrupt', 'count']
    bankrupt_counts['Bankrupt'] = bankrupt_counts['Bankrupt'].map({0: 'Non-Bankrupt', 1: 'Bankrupt'})
    fig = px.pie(bankrupt_counts, values='count', names='Bankrupt',
                 title='Proportion of Bankrupt vs. Non-Bankrupt Companies',
                 color_discrete_map={'Non-Bankrupt': 'rgb(26, 118, 255)', 'Bankrupt': 'rgb(255, 127, 14)'})
    return fig

# --- NEW FOOTER FUNCTION ---
def add_footer():
    """Adds a custom footer to the page."""
    footer_css = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117; /* Matches Streamlit's dark theme background */
        color: #FAFAFA; /* Light text color */
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #262730; /* A subtle top border */
    }
    </style>
    """
    footer_html = '<div class="footer"><p>@2025</p></div>'
    st.markdown(footer_css, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)

def app():
    """This function contains the content for the visualization page."""
    # The st.set_page_config() call was removed from here.
    st.title("Bankruptcy Prediction Dataset Visualizer")
    st.markdown("Use this app to explore the financial data and visualize key relationships for bankruptcy prediction.")

    df = load_data()

    if df is not None:
        # Get a list of all numerical columns, excluding the 'Bankrupt' target
        numerical_cols = [col for col in df.columns if col != 'Bankrupt' and df[col].dtype in ['float64', 'int64']]

        st.sidebar.header("Select a Visualization")
        visualization_type = st.sidebar.selectbox(
            "Choose a plot type:",
            ["Pie Chart", "Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"]
        )

        if visualization_type == "Histogram":
            st.header("Histogram")
            st.markdown(
                "A histogram shows the distribution of a single numerical variable. "
                "The app plots the histogram with different colors for bankrupt and non-bankrupt "
                "companies, allowing you to visually compare the distribution of a financial ratio "
                "between the two groups."
            )
            column = st.selectbox("Select a feature to visualize:", numerical_cols)
            if column:
                st.plotly_chart(create_histogram(df, column), use_container_width=True)

        elif visualization_type == "Box Plot":
            st.header("Box Plot")
            st.markdown(
                "A box plot provides a summary of the distribution of a variable and is useful for "
                "comparing distributions across different categories (bankrupt vs. non-bankrupt). "
                "The median, interquartile range (IQR), and outliers are clearly visible, "
                "helping you see if there is a significant difference in a ratio between the two groups."
            )
            column = st.selectbox("Select a feature to compare:", numerical_cols)
            if column:
                st.plotly_chart(create_box_plot(df, column), use_container_width=True)

        elif visualization_type == "Scatter Plot":
            st.header("Scatter Plot")
            st.markdown(
                "A scatter plot displays the relationship between two numerical variables. "
                "The points are colored by their bankruptcy status. Look for distinct clusters "
                "of bankrupt and non-bankrupt companies, which indicates that the two features "
                "together are good predictors."
            )
            col1 = st.selectbox("Select X-axis feature:", numerical_cols)
            col2 = st.selectbox("Select Y-axis feature:", [c for c in numerical_cols if c != col1])
            if col1 and col2:
                st.plotly_chart(create_scatter_plot(df, col1, col2), use_container_width=True)

        elif visualization_type == "Correlation Heatmap":
            st.header("Correlation Heatmap")
            st.markdown(
                "A correlation heatmap displays the correlation coefficients between all pairs of numerical "
                "variables. The color of each cell represents the strength and direction of the "
                "correlation. This provides a quick overview of the entire dataset and helps you "
                "identify which features are most strongly related."
            )
            st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)

        elif visualization_type == "Pie Chart":
            st.header("Class Distribution Pie Chart")
            st.markdown(
                "A pie chart is used to visualize the composition of a categorical variable. "
                "In this case, it shows the proportion of companies that are bankrupt versus "
                "those that are non-bankrupt in the dataset. This is crucial for understanding class imbalance."
            )
            st.plotly_chart(create_pie_chart(df), use_container_width=True)

        # --- CALL THE FOOTER FUNCTION HERE ---
        add_footer()

# --- Entry point to run the app ---
if __name__ == '__main__':
    app()

