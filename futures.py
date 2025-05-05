import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def futures_definition():
    st.title("Understanding Futures Contracts")

    st.markdown("""
    ### 📘 **Futures Contracts: A Primer**

    Futures contracts are standardized agreements to buy or sell a specific quantity of an asset at a predetermined price on a set future date. 
    They are traded on organized exchanges and are essential tools for **hedging** and **speculation** in financial markets.
    """)

    st.markdown("""
    ### 🔍 **Key Concepts of Futures Contracts**

    - **Standardization:** All futures contracts are standardized in terms of size, expiration date, and delivery terms, facilitating liquidity and exchange trading.
    - **Margin and Marking to Market:** 
        - Investors deposit an **initial margin** to open a position.
        - **Marking to market** adjusts account balances daily based on gains/losses.
        - A **maintenance margin** ensures sufficient equity is held; falling below triggers a margin call.
    - **Settlement Types:**
        - **Physical Delivery:** Actual asset is delivered (e.g., oil, corn).
        - **Cash Settlement:** No delivery occurs; the difference in contract value is exchanged.
    - **Long vs. Short:**
        - **Long Position:** Agree to buy the asset at contract maturity.
        - **Short Position:** Agree to sell the asset at contract maturity.
    """)

    st.markdown("""
    ### 📈 **Uses of Futures**

    1. **Hedging:**  
       Futures allow businesses and investors to lock in prices and offset risks.  
       *Example:* An airline hedges fuel cost risk using oil futures.

    2. **Speculation:**  
       Traders use futures to bet on the direction of market prices without owning the underlying asset.

    3. **Arbitrage:**  
       Exploiting price differences between futures and spot markets or between different exchanges.
    """)

    st.markdown("""
    ### ⚙️ **Common Futures Contracts**

    | Asset Class        | Example Contract     | Exchange           |
    |--------------------|----------------------|--------------------|
    | Commodities        | Crude Oil, Gold      | NYMEX, COMEX       |
    | Financial Indexes  | S&P 500 Futures      | CME Group          |
    | Currencies         | EUR/USD Futures      | CME                |
    | Interest Rates     | Treasury Futures     | CBOT               |
    """)

    st.markdown("""
    ### 🧠 **Why Learn Futures?**

    Mastering futures helps investors manage risk, speculate effectively, and understand market sentiment. 
    Futures are central to many **portfolio strategies**, **hedging models**, and **derivative pricing frameworks**.
    """)

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
    page = st.sidebar.selectbox("Select Page", ["Futures Description", "Oil Hedging", "Oil Hedging 2"])

    if "page" not in st.session_state:
        st.session_state.page = "Futures Description"
    else:
        st.session_state.page = page

    if st.session_state.page == "Futures Description":
        futures_definition()
    elif st.session_state.page == "Oil Hedging":
        oil_hedging()
    elif st.session_state.page == "Oil Hedging 2":
        oil_hedging()

if __name__ == "__main__":
    app()
