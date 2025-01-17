import streamlit as st
from stocks import app as stocks_app
from bonds import app as bonds_app
from portfolio_theory import app as portfolio_theory_app
from options import app as options_app

NAVIGATION_BAR_STYLE = """
    <style>
        /* Adjust the sidebar background color */
        section[data-testid="stSidebar"] {
            background-color : #66A2C4 ;
        }
        /* Adjust padding and margin for sidebar elements */
        section[data-testid="stSidebar"] .css-1lcbmhc {
            padding-top: 2rem !important;
            margin-top: 1rem !important;
        }
        /* Style the navigation items */
        section[data-testid="stSidebar"] .css-1lcbmhc .css-qrbaxs {
            padding: 0.5rem 1rem !important;
            border-radius: 5px !important;
            margin-bottom: 0.5rem !important;
        }
        section[data-testid="stSidebar"] .css-1lcbmhc .css-qrbaxs a {
            color: #000000 !important;
            text-decoration: none !important;
            display: block !important;
            padding: 0.5rem 1rem !important;
            border-radius: 5px !important;
            margin-bottom: 0.5rem !important;
            transition: background-color 0.3s, color 0.3s !important;
        }
        section[data-testid="stSidebar"] .css-1lcbmhc .css-qrbaxs a:hover {
            background-color: #007bff !important;
            color: #ffffff !important;
        }
    </style>
"""

def main():
    st.set_page_config(
        page_title="Finance App",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )

    # Customizing the sidebar with CSS
    st.sidebar.markdown(NAVIGATION_BAR_STYLE, unsafe_allow_html=True)
    st.sidebar.title('Navigation')

    # Create a dropdown menu for pages
    page_selection = st.sidebar.selectbox(
        "Go to",
        ["Home", "Stocks", "Bonds", "Portfolio Theory", "Options"]
    )

    # Display content based on selection
    if page_selection == "Home":
        st.title("Welcome to the FinTech App")
        st.write("This is the main page. Customize it as per your needs.")
    elif page_selection == "Stocks":
        stocks_app()
    elif page_selection == "Bonds":
        bonds_app()
    elif page_selection == "Portfolio Theory":
        portfolio_theory_app()
    elif page_selection == "Options":
        options_app()

if __name__ == '__main__':
    main()
