import streamlit as st
from stocks import app as stocks_app
from bonds import app as bonds_app
from portfolio_theory import app as portfolio_theory_app
from options import app as options_app
from market import app as market_app
from futures import app as futures_app

NAVIGATION_BAR_STYLE = """
    <style>
        section[data-testid="stSidebar"] {
            background-color : #66A2C4 ;
        }
        section[data-testid="stSidebar"] .css-1lcbmhc {
            padding-top: 2rem !important;
            margin-top: 1rem !important;
        }
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
        page_title="FinTech Learning App",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )

    st.sidebar.markdown(NAVIGATION_BAR_STYLE, unsafe_allow_html=True)
    st.sidebar.title('Navigation')

    page_selection = st.sidebar.selectbox(
        "Go to",
        ["Home", "Stocks", "Bonds", "Portfolio Theory", "Options", "Market Efficiency", "Futures"]
    )

    if page_selection == "Home":
        st.title("📈 Welcome to the FinTech Learning App")
        st.markdown("""
        **Empowering financial understanding through interactive exploration.**

        This application is built as an educational companion to *Essentials of Investments* by Bodie, Kane, and Marcus — one of the most widely used texts in finance education. It transforms key investment concepts into intuitive, visual, and interactive experiences.
        """)

        st.markdown("### 🧭 What You’ll Explore:")
        st.markdown("""
        #### 🏦 **Stocks**
        - Dividend Discount Model (DDM)
        - Free Cash Flow to Firm (FCFF)
        - Drivers of stock price and valuation
        - Growth, risk, and market scenarios

        #### 💸 **Bonds**
        - Yield vs. price relationship
        - Duration, convexity, and yield curves
        - Interactive bond cash flow modeling

        #### 🧠 **Portfolio Theory**
        - Diversification, Efficient Frontier, and Capital Market Line
        - Sharpe Ratio, historical return distributions
        - Growth of $1, volatility and risk-adjusted return plots

        #### 📊 **Options**
        - Black-Scholes and binomial pricing models
        - Visual strategies: protective puts, covered calls
        - Intrinsic and time value breakdowns

        #### 📉 **Market Efficiency**
        - Efficient Market Hypothesis (EMH)
        - Anomalies and behavioral finance
        - Historical inflation vs. T-bills, S&P 500 distributions

        #### 🔁 **Futures**
        - Futures pricing, hedging, and speculation
        - Margining mechanics
        - Use cases for commodities, rates, and equity indices
        """)

        st.markdown("### 📚 Built for Learning and Practice")
        st.markdown("""
        Each module includes:
        - 📊 **Interactive Graphs**
        - 🧪 **Simulations**
        - 🧠 **Theory + Data Integration**
        - 📈 **Upload and analyze your own datasets**
        """)

        st.markdown("### 🎯 Who is This For?")
        st.markdown("""
        - **Students** in finance, economics, or MBA programs
        - **Investors** seeking to deepen their understanding
        - **Educators** teaching core investment principles
        - **Career switchers** entering the financial or data-driven investing world
        """)

        st.markdown("### 🔐 Powered By:")
        st.markdown("""
        - Python, Streamlit, Pandas, Matplotlib
        - Data APIs: FRED, Yahoo Finance
        - Based on *Essentials of Investments* by Bodie, Kane, and Marcus
        """)
    
    elif page_selection == "Stocks":
        stocks_app()
    elif page_selection == "Bonds":
        bonds_app()
    elif page_selection == "Portfolio Theory":
        portfolio_theory_app()
    elif page_selection == "Options":
        options_app()
    elif page_selection == "Market Efficiency":
        market_app()
    elif page_selection == "Futures":
        futures_app()

if __name__ == '__main__':
    main()
