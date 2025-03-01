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
        task()


if __name__ == "__main__":
    app()