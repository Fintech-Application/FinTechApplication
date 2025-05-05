import streamlit as st
from stocks import app as stocks_app
from bonds import app as bonds_app
from portfolio_theory import app as portfolio_theory_app
from options import app as options_app
from market import app as market_app
from futures import app as futures_app

# Sidebar CSS styling
NAVIGATION_BAR_STYLE = """
<style>
section[data-testid="stSidebar"] {
    background-color: #66A2C4;
}
section[data-testid="stSidebar"] .css-1lcbmhc {
    padding-top: 2rem !important;
    margin-top: 1rem !important;
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

# Home Page Function
def homepage():
    st.title("📊 FinTech Learning Hub")
    st.markdown("Welcome to your interactive playground for mastering Financial Markets and Investment Analytics.")

    st.markdown("""
    This app is designed for students, professionals, and curious minds to **learn, simulate, and visualize** key financial concepts using:
    - 🧮 Mathematical Models
    - 📈 Real-world Data
    - 💡 Intuitive Visualizations
    - 🧠 Practical Insights
    """)

    st.subheader("🚀 What You'll Learn")
    st.markdown("""
    - 📘 **Options**: Black-Scholes, Binomial Trees, Put-Call Parity, Covered Calls, Protective Puts  
    - 📗 **Portfolio Theory**: Efficient Frontier, Sharpe Ratio, Growth of $1, Risk/Return  
    - 📙 **Bonds & T-Bills**: Inflation vs. T-Bills, Frequency Distributions  
    - 📕 **Market Data**: S&P 500 Trends, Volatility, Sharpe Ratios  
    - 📒 **Futures & Market Efficiency**: Strategy insights and risk analysis  
    """)

    st.markdown("Use the **sidebar** to start exploring each module.")
    st.markdown("---")
    st.markdown("Built by **Seyi Swathhy Yaganti** | [LinkedIn](https://www.linkedin.com/in/swathhy-yaganti/)")

# Main App Controller
def main():
    st.set_page_config(page_title="FinTech App", page_icon=":chart_with_upwards_trend:", layout="wide")

    # Inject custom sidebar styling
    st.sidebar.markdown(NAVIGATION_BAR_STYLE, unsafe_allow_html=True)
    st.sidebar.title('🔎 Navigation')

    # Page selector
    page = st.sidebar.selectbox(
        "Select Module",
        ["Home", "Stocks", "Bonds", "Portfolio Theory", "Options", "Market Efficiency", "Futures"]
    )

    # Page routing logic
    if page == "Home":
        homepage()
    elif page == "Stocks":
        stocks_app()
    elif page == "Bonds":
        bonds_app()
    elif page == "Portfolio Theory":
        portfolio_theory_app()
    elif page == "Options":
        options_app()
    elif page == "Market Efficiency":
        market_app()
    elif page == "Futures":
        futures_app()

# Launch
if __name__ == '__main__':
    main()
