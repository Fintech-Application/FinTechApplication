import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def market_efficiency_definition():
    st.write("this is market efficiency")

def task():
    st.write("this is task")
    
# Streamlit UI
def app():
    st.title('Market Efficiency Page')
    page = st.sidebar.selectbox("Select Page", ["Market Efficiency Description", "Task"])

    # Track the current page
    if "page" not in st.session_state:
        st.session_state.page = "Market Efficiency Description"
    else:
        st.session_state.page = page

    if st.session_state.page == "Market Efficiency Description":
        market_efficiency_definition()

    elif st.session_state.page == "Task":
        general_dividend_discount_model()

    elif st.session_state.page == "Constant Dividend Growth Model":
        constant_dividend_growth_model()

    elif st.session_state.page == "Two Stage Dividend Growth Model":
        two_stage_growth_model()
    
    elif st.session_state.page == "FCFF":
        # Ask the user to input FCFF values for each year in the high-growth phase
        fcff_list = []
        for i in range(1, 6):  # Assuming high-growth phase is 5 years
            fcff_value = st.number_input(f"Enter FCFF for Year {i} (in million):", min_value=0.0, format="%.2f")
            fcff_list.append(fcff_value)
        
        FCFF(fcff_list)  # Pass the fcff_list to FCFF
    
    elif st.session_state.page == "Free Cash Flows":
        free_cash_flows()
    
    elif st.session_state.page == "FCFF webscraping TopDown":
        FCFF_webscraping_TopDown()
    
    elif st.session_state.page == "FCFF webscraping BottomUp":
        FCFF_webscraping_BottomUp()

if __name__ == "__main__":
    app()