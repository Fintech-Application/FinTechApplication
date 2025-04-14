import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def futures_definition():
    # Title
    st.title("Understanding Futures")

def oil_hedging():
    st.title("Oil Price Hedging Simulator")
    st.write("""
    Adjust the parameters below to simulate different hedging scenarios for oil prices.
    """)

    # User Inputs
    num_barrels = st.number_input("Number of Barrels", min_value=1, value=100000, step=10000)
    futures_price = st.number_input("Futures Price (Fo) ($)", min_value=1, value=52, step=1)

    # Contract size is fixed
    contract_size = 1000  
    auto_num_contracts = num_barrels // contract_size  # Automatically calculate contracts
    
    # User input for number of contracts
    num_contracts = st.number_input("Number of Contracts", min_value=1, value=auto_num_contracts, step=1)

    # Oil price range inputs
    oil_price_min = st.number_input("Minimum Oil Price (P) ($)", min_value=1, value=51, step=1)
    oil_price_max = st.number_input("Maximum Oil Price (P) ($)", 
                                    min_value=oil_price_min + 1, 
                                    value=max(oil_price_min + 2, oil_price_min + 1),
                                    step=1)
    
    # Define oil prices dynamically
    prices = np.arange(oil_price_min, oil_price_max + 1, 1)

    # Compute proceeds per barrel
    revenue_per_barrel = prices  # Sales revenue per barrel
    futures_profit_per_barrel = futures_price - prices  # Profit on futures per barrel
    total_proceeds_per_barrel = revenue_per_barrel + futures_profit_per_barrel  # Total proceeds per barrel

    # Display number of contracts used for hedging
    st.write(f"**Number of Contracts Used:** {num_contracts} (Each contract = {contract_size} barrels)")

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(prices, revenue_per_barrel, label="Sales Revenue per Barrel", marker="o", linestyle="-")
    ax.plot(prices, futures_profit_per_barrel, label="Futures Profits per Barrel", marker="o", linestyle="-")
    ax.plot(prices, total_proceeds_per_barrel, label="Total Proceeds per Barrel", marker="o", linestyle="--")

    # Set dynamic Y-axis limits based on the computed values
    y_min = min(np.min(revenue_per_barrel), np.min(futures_profit_per_barrel), np.min(total_proceeds_per_barrel)) - 5
    y_max = max(np.max(revenue_per_barrel), np.max(futures_profit_per_barrel), np.max(total_proceeds_per_barrel)) + 5

    ax.set_ylim(y_min, y_max)  # Dynamically adjust the Y-axis range

    # Axis labels and title
    ax.set_xlabel("Oil Price in February ($)")
    ax.set_ylabel("Proceeds per Barrel ($)")
    ax.set_title("Hedging Revenues Using Futures")

    ax.legend()
    ax.grid(True)

    # Show plot in Streamlit
    st.pyplot(fig)

def app():
    page = st.sidebar.selectbox("Select Page", ["Futures Description", "Oil Hedging"])

    if "page" not in st.session_state:
        st.session_state.page = "Futures Description"
    else:
        st.session_state.page = page

    if st.session_state.page == "Futures Description":
        futures_definition()
    elif st.session_state.page == "Oil Hedging":
        oil_hedging()

if __name__ == "__main__":
    app()
