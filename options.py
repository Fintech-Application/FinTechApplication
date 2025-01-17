import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Black-Scholes model functions
def d1(Stockprice, exerciseprice, maturity, riskfreerate, sigma):
    return (log(Stockprice / exerciseprice) + (riskfreerate + sigma ** 2 / 2) * maturity) / (sigma * sqrt(maturity))

def d2(Stockprice, exerciseprice, maturity, riskfreerate, sigma):
    return d1(Stockprice, exerciseprice, maturity, riskfreerate, sigma) - sigma * sqrt(maturity)

def bs_call(Stockprice, exerciseprice, maturity, riskfreerate, sigma):
    return Stockprice * norm.cdf(d1(Stockprice, exerciseprice, maturity, riskfreerate, sigma)) - exerciseprice * exp(-riskfreerate * maturity) * norm.cdf(d2(Stockprice, exerciseprice, maturity, riskfreerate, sigma))

def bs_put(Stockprice, exerciseprice, maturity, riskfreerate, sigma):
    return exerciseprice * exp(-riskfreerate * maturity) * norm.cdf(-d2(Stockprice, exerciseprice, maturity, riskfreerate, sigma)) - Stockprice * norm.cdf(-d1(Stockprice, exerciseprice, maturity, riskfreerate, sigma))

def intrinsic_payoff(S, X):
    return np.maximum(0, S - X)

def static_call_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    pricearray = [bs_call(S, exerciseprice, maturity, riskfreerate, sigma) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, pricearray, label="Call Option Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Call Option Price")
    ax.set_title("Black-Scholes Call Option Price vs Stock Price")
    ax.legend()
    ax.grid(True)

    return fig

def static_put_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    pricearray = [bs_put(S, exerciseprice, maturity, riskfreerate, sigma) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, pricearray, label="Put Option Price", color='red')
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Put Option Price")
    ax.set_title("Black-Scholes Put Option Price vs Stock Price")
    ax.legend()
    ax.grid(True)

    return fig

#dynamic graph
def call_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    pricearray = [bs_call(S, exerciseprice, maturity, riskfreerate, sigma) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, pricearray, label="Call Option Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Call Option Price")
    ax.set_title("Black-Scholes Call Option Price vs Stock Price")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=-50, top=300)

    return fig

def put_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    pricearray = [bs_put(S, exerciseprice, maturity, riskfreerate, sigma) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, pricearray, label="Put Option Price", color='red')
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Put Option Price")
    ax.set_title("Black-Scholes Put Option Price vs Stock Price")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=-50, top=300)

    return fig

#comparison of dynamic graphs
def compare_call_option_vs_stock_price(exerciseprice1, maturity1, riskfreerate1, sigma1,
                               exerciseprice2, maturity2, riskfreerate2, sigma2,
                               exerciseprice3, maturity3, riskfreerate3, sigma3):
    Stockprice_min = 50
    Stockprice_max = 500
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    
    # Compute call option prices for each parameter set
    pricearray1 = [bs_call(S, exerciseprice1, maturity1, riskfreerate1, sigma1) for S in rangelist]
    pricearray2 = [bs_call(S, exerciseprice2, maturity2, riskfreerate2, sigma2) for S in rangelist]
    pricearray3 = [bs_call(S, exerciseprice3, maturity3, riskfreerate3, sigma3) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, pricearray1, label=f"Set 1: Exercise Price = {exerciseprice1}, Maturity = {maturity1}, Risk-Free Rate = {riskfreerate1}, Sigma = {sigma1}")
    ax.plot(rangelist, pricearray2, label=f"Set 2: Exercise Price = {exerciseprice2}, Maturity = {maturity2}, Risk-Free Rate = {riskfreerate2}, Sigma = {sigma2}")
    ax.plot(rangelist, pricearray3, label=f"Set 3: Exercise Price = {exerciseprice3}, Maturity = {maturity3}, Risk-Free Rate = {riskfreerate3}, Sigma = {sigma3}")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Call Option Price")
    ax.set_title("Black-Scholes Call Option Price vs Stock Price")
    ax.legend()
    ax.grid(True)

    return fig

def compare_put_option_vs_stock_price(exerciseprice1, maturity1, riskfreerate1, sigma1,
                                   exerciseprice2, maturity2, riskfreerate2, sigma2,
                                   exerciseprice3, maturity3, riskfreerate3, sigma3):
    Stockprice_min = 50
    Stockprice_max = 500
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    
    # Compute put option prices for each parameter set
    pricearray1 = [bs_put(S, exerciseprice1, maturity1, riskfreerate1, sigma1) for S in rangelist]
    pricearray2 = [bs_put(S, exerciseprice2, maturity2, riskfreerate2, sigma2) for S in rangelist]
    pricearray3 = [bs_put(S, exerciseprice3, maturity3, riskfreerate3, sigma3) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, pricearray1, label=f"Set 1: Exercise Price = {exerciseprice1}, Maturity = {maturity1}, Risk-Free Rate = {riskfreerate1}, Sigma = {sigma1}")
    ax.plot(rangelist, pricearray2, label=f"Set 2: Exercise Price = {exerciseprice2}, Maturity = {maturity2}, Risk-Free Rate = {riskfreerate2}, Sigma = {sigma2}")
    ax.plot(rangelist, pricearray3, label=f"Set 3: Exercise Price = {exerciseprice3}, Maturity = {maturity3}, Risk-Free Rate = {riskfreerate3}, Sigma = {sigma3}")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Put Option Price")
    ax.set_title("Black-Scholes Put Option Price vs Stock Price")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=-50, top=300)  # Set y-axis limit with gap below 0

    return fig

#put call parity
#graph-1
def plot_call_option_price_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    call_prices = [bs_call(S, exerciseprice, maturity, riskfreerate, sigma) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, call_prices, label="Call Option Price (Ct)", color='blue')
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Call Option Price (Ct)")
    ax.set_title("Call Option Price (Ct) vs Stock Price (S)")
    ax.legend()
    ax.grid(True)

    return fig
#graph-2
def plot_present_value_of_exercise_price(exerciseprice, maturity, riskfreerate):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    present_value = [exerciseprice / (1 + riskfreerate) ** maturity for _ in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, present_value, label="Present Value of Exercise Price (X / (1+r)^T)", color='green')
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Present Value (X / (1+r)^T)")
    ax.set_title("Present Value of Exercise Price vs Stock Price (S)")
    ax.legend()
    ax.grid(True)

    return fig

# Combined Graph
def plot_put_call_parity_graph(exerciseprice, maturity, riskfreerate, sigma):
    Stockprice_min = 50  # Set minimum stock price for range
    Stockprice_max = 500  # Set maximum stock price for range
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    
    lhs = [bs_call(S, exerciseprice, maturity, riskfreerate, sigma) + exerciseprice / (1 + riskfreerate) ** maturity for S in rangelist]
    rhs = [S + bs_put(S, exerciseprice, maturity, riskfreerate, sigma) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, lhs, label="LHS: Ct + X / (1+r)^T", linestyle='--', color='blue')
    ax.plot(rangelist, rhs, label="RHS: S + Pt", linestyle='-', color='red')
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Value")
    ax.set_title("Put-Call Parity at Maturity")
    ax.legend()
    ax.grid(True)

    return fig

#intrinsic payoff graph
def intrinsic_payoff_vs_stock_price(exerciseprice):
    Stockprice_min = 50
    Stockprice_max = 500
    rangelist = np.linspace(Stockprice_min, Stockprice_max, 400)
    
    # Compute intrinsic payoffs for the given exercise price
    payoffarray = [intrinsic_payoff(S, exerciseprice) for S in rangelist]

    fig, ax = plt.subplots()
    ax.plot(rangelist, payoffarray, label=f"Exercise Price = {exerciseprice}")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Intrinsic Payoff")
    ax.set_title("Intrinsic Payoff vs Stock Price")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=-50, top=300) 

    return fig


def calculate_option_price(S0, U, d, X, T, rf):
    r = rf / 100  # Convert rate to decimal
    dt = 1        # Assuming single time step for simplicity
    q = (1 + r - d) / (U - d)  # Risk-neutral probability
    disc = 1 / (1 + r)         # Discount factor
    
    # Stock prices at expiration
    S_up = S0 * U
    S_down = S0 * d

    # Option payoffs at expiration
    payoff_up = max(S_up - X, 0)  # Call option payoff
    payoff_down = max(S_down - X, 0)
    
    # Option value using risk-neutral valuation
    C = disc * (q * payoff_up + (1 - q) * payoff_down)

    # Hedge ratio
    hedge_ratio = (payoff_up - payoff_down) / (S_up - S_down)

    return C, hedge_ratio, S_up, S_down, payoff_up, payoff_down

def binomial_option_pricing(S0, U, d, C, S_up, S_down, payoff_up, payoff_down):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Stock Price Tree
    axs[0].plot([0, 1], [S0, S_up], 'ro-', label='Stock Price (Up)')
    axs[0].plot([0, 1], [S0, S_down], 'bo-', label='Stock Price (Down)')
    axs[0].text(0, S0, f"${S0:.2f}", fontsize=10, color='black', ha='center')
    axs[0].text(1, S_up, f"${S_up:.2f}", fontsize=10, color='green', ha='center')
    axs[0].text(1, S_down, f"${S_down:.2f}", fontsize=10, color='red', ha='center')
    axs[0].set_title("Stock Price Tree")
    axs[0].set_ylabel("Stock Price")
    axs[0].grid(False)

    # Option Value Tree (Node-and-Edge Structure)
    # Plot C node
    axs[1].plot(1, C, 'bo', label=f'Call Option Value (C): ${C:.2f}')
    axs[1].text(1, C, f"C", fontsize=14, color='blue', ha='center', va='center')
    
    # Plot branches and Up/Down values
    axs[1].plot([1, 2], [C, payoff_up], 'go-', label='Option Value (Up)')
    axs[1].plot([1, 2], [C, payoff_down], 'yo-', label='Option Value (Down)')
    
    # Label the option values at the next level
    axs[1].text(2, payoff_up, f"${payoff_up:.2f}", fontsize=10, color='green', ha='center')
    axs[1].text(2, payoff_down, f"${payoff_down:.2f}", fontsize=10, color='red', ha='center')

    axs[1].set_title("Option Value Tree")
    axs[1].set_ylabel("Option Value")
    axs[1].grid(False)

    # Adjust layout and legend
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()

    return fig

def app():
    st.title("Option Analysis")

    # Sidebar for page navigation
    page = st.sidebar.selectbox("Select Page", ["Options Description","Static Call Option Price vs Stock Price", "Static Put Option Price vs Stock Price", "Call Option Price vs Stock Price", "Put Option Price vs Stock Price", "Compare Call option prices","Compare Put option prices" ,"Put-Call Parity", "Intrinsic Payoff", "Binomial Option Pricing"])

    # Track the current page
    if "page" not in st.session_state:
        st.session_state.page = "Options Description"
    else:
        st.session_state.page = page

    if st.session_state.page == "Options Description":
        st.write("Options are financial derivatives that give buyers the right, but not the obligation, to buy or sell an underlying asset at an agreed-upon price and date.")
        # Add more descriptive content here

    elif st.session_state.page == "Static Call Option Price vs Stock Price":
        st.title('Call Option Price vs Stock Price Graph')
        exerciseprice = 200  # Example Exercise Price
        maturity = 1         # Example Maturity (1 year)
        riskfreerate = 0.05  # Example Risk-Free Rate (5%)
        sigma = 0.2          # Example Volatility (20%)

        # Generate and display the static graph
        fig = static_call_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma)
        st.pyplot(fig)
        st.markdown("""### Definition of Call Option
        A call option gives the holder the right, but not the obligation, to buy an underlying asset at a predetermined strike price (exercise price) before or at expiration.

        ### Black-Scholes Model Overview
        The Black-Scholes model calculates the theoretical price of European-style call and put options based on factors such as stock price, strike price, time to maturity, risk-free rate, and volatility.

        ### Key Inputs
        - **Exercise Price (Strike Price)**: The price at which the underlying asset can be bought.
        - **Stock Price**: The current price of the underlying asset.
        - **Maturity**: The time remaining until the option expires.
        - **Risk-Free Rate**: The theoretical return on a risk-free investment, typically represented by government treasury yields.
        - **Volatility**: A measure of the asset’s price fluctuation, indicating the uncertainty of the asset's return.

        ### Graph Interpretation
        - **X-Axis (Stock Price)**: Shows the range of possible stock prices at expiration.
        - **Y-Axis (Call Option Price)**: Represents the price of the call option calculated using the Black-Scholes formula.
        - As stock price increases, the call option price typically increases, reflecting the greater potential value of the option.

        ### Call Option Pricing Behavior
        - **In-the-Money**: When the stock price is above the exercise price, the call option price increases. The intrinsic value is positive.
        - **At-the-Money**: When the stock price is equal to the exercise price, the call option price is influenced primarily by time value and volatility.
        - **Out-of-the-Money**: When the stock price is below the exercise price, the call option price is lower and approaches zero as the stock price decreases.

        ### Effects of Parameters
        - **Higher Exercise Price**: Decreases the call option price for a given stock price, as the option becomes less likely to be exercised profitably.
        - **Longer Maturity**: Increases the call option price, as more time allows for a greater chance of the stock price exceeding the exercise price.
        - **Higher Risk-Free Rate**: Increases the call option price, as the present value of the exercise price is reduced.
        - **Higher Volatility**: Increases the call option price, as greater price fluctuations enhance the potential value of the option.""")
            
    elif st.session_state.page == "Static Put Option Price vs Stock Price":
        st.title('Put Option Price vs Stock Price Graph')
        exerciseprice = 200  # Example Exercise Price
        maturity = 1         # Example Maturity (1 year)
        riskfreerate = 0.05  # Example Risk-Free Rate (5%)
        sigma = 0.2          # Example Volatility (20%)

        # Generate and display the static graph
        fig = static_put_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma)
        st.pyplot(fig)
        st.markdown("""### Definition of Put Option
                
        A put option gives the holder the right, but not the obligation, to sell an underlying asset at a predetermined strike price (exercise price) before or at expiration.

        ### Black-Scholes Model Overview
        The Black-Scholes model calculates the theoretical price of European-style call and put options based on factors such as stock price, strike price, time to maturity, risk-free rate, and volatility.

        ### Key Inputs
        - **Exercise Price (Strike Price)**: The price at which the underlying asset can be sold.
        - **Stock Price**: The current price of the underlying asset.
        - **Maturity**: The time remaining until the option expires.
        - **Risk-Free Rate**: The theoretical return on a risk-free investment, typically represented by government treasury yields.
        - **Volatility**: A measure of the asset’s price fluctuation, indicating the uncertainty of the asset's return.

        ### Graph Interpretation
        - **X-Axis (Stock Price)**: Shows the range of possible stock prices at expiration.
        - **Y-Axis (Put Option Price)**: Represents the price of the put option calculated using the Black-Scholes formula.
        - As stock price decreases, the put option price typically increases, reflecting the greater potential value of the option.

        ### Put Option Pricing Behavior
        - **In-the-Money**: When the stock price is below the exercise price, the put option price increases. The intrinsic value is positive.
        - **At-the-Money**: When the stock price is equal to the exercise price, the put option price is influenced primarily by time value and volatility.
        - **Out-of-the-Money**: When the stock price is above the exercise price, the put option price is lower and approaches zero as the stock price increases.

        ### Effects of Parameters
        - **Higher Exercise Price**: Increases the put option price for a given stock price, as the option becomes more likely to be exercised profitably.
        - **Longer Maturity**: Increases the put option price, as more time allows for a greater chance of the stock price falling below the exercise price.
        - **Higher Risk-Free Rate**: Decreases the put option price, as the present value of the exercise price is reduced.
        - **Higher Volatility**: Increases the put option price, as greater price fluctuations enhance the potential value of the option.
        """)

    elif st.session_state.page == "Call Option Price vs Stock Price":
        st.title('Dynamic Visualization of Call Option Price vs. Stock Price')
        st.sidebar.write("Adjust parameters for the Black-Scholes model:")
        exerciseprice = st.sidebar.slider("Exercise Price", 50, 500, 330)
        maturity = st.sidebar.slider("Maturity (in years)", 0.01, 2.0, 0.25, 0.01)
        riskfreerate = st.sidebar.slider("Risk-Free Rate", 0.01, 0.10, 0.04)
        sigma = st.sidebar.slider("Volatility", 0.1, 1.0, 0.4029, 0.01)
        fig = call_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma)
        st.pyplot(fig)
    
    elif st.session_state.page == "Put Option Price vs Stock Price":
        st.title('Dynamic Visualization of Put Option Price vs. Stock Price')
        st.sidebar.write("Adjust parameters for the Black-Scholes model:")
        exerciseprice = st.sidebar.slider("Exercise Price", 50, 500, 330)
        maturity = st.sidebar.slider("Maturity (in years)", 0.1, 2.0, 0.25, 0.01)
        riskfreerate = st.sidebar.slider("Risk-Free Rate", 0.01, 0.10, 0.04)
        sigma = st.sidebar.slider("Volatility", 0.1, 1.0, 0.4029, 0.01)
        
        # Generate the put option price vs. stock price graph
        fig = put_option_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma)
        st.pyplot(fig)
    
    elif st.session_state.page == "Compare Call option prices":
        st.title('Comparative Analysis of Call Option Price vs. Stock Price Graphs')
        col1, col2, col3 = st.columns(3)
        exerciseprice1 = col1.slider("Exercise Price 1", min_value=10, max_value=500, value=200)
        maturity1 = col1.slider("Maturity 1 (years)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        riskfreerate1 = col1.slider("Risk-Free Rate 1", min_value=0.0, max_value=0.2, step=0.01, value=0.05)
        sigma1 = col1.slider("Volatility 1", min_value=0.01, max_value=1.0, step=0.01, value=0.2)

        # Sliders for parameter set 2
        exerciseprice2 = col2.slider("Exercise Price 2", min_value=10, max_value=500, value=250)
        maturity2 = col2.slider("Maturity 2 (years)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        riskfreerate2 = col2.slider("Risk-Free Rate 2", min_value=0.0, max_value=0.2, step=0.01, value=0.05)
        sigma2 = col2.slider("Volatility 2", min_value=0.01, max_value=1.0, step=0.01, value=0.3)

        # Sliders for parameter set 3
        exerciseprice3 = col3.slider("Exercise Price 3", min_value=10, max_value=500, value=300)
        maturity3 = col3.slider("Maturity 3 (years)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        riskfreerate3 = col3.slider("Risk-Free Rate 3", min_value=0.0, max_value=0.2, step=0.01, value=0.05)
        sigma3 = col3.slider("Volatility 3", min_value=0.01, max_value=1.0, step=0.01, value=0.4)

        # Generate and display the dynamic graph
        fig = compare_call_option_vs_stock_price(exerciseprice1, maturity1, riskfreerate1, sigma1,
                                        exerciseprice2, maturity2, riskfreerate2, sigma2,
                                        exerciseprice3, maturity3, riskfreerate3, sigma3)
        st.pyplot(fig)
        st.markdown("""
        ### Key Insights from Comparing Call Option Price vs. Stock Price Graphs

        1. **Exercise Price Impact**:  
        Higher exercise prices lower call option values as they reduce the likelihood of profitability. Comparing different exercise prices shows the sensitivity of options to strike price variations.

        2. **Maturity Effect**:  
        Longer maturities increase call option prices due to higher time value, reflecting more time for the stock to move favorably. This highlights the diminishing time value as expiration nears.

        3. **Risk-Free Rate Influence**:  
        A higher risk-free rate increases call option prices since future payoffs are discounted less. Changes in this rate can significantly affect option values, especially in low-interest environments.

        4. **Volatility's Role**:  
        Increased volatility raises option prices by enhancing the probability of profitable outcomes. Graphs with varying volatilities demonstrate the "insurance" value of options amid market uncertainty.

        5. **Market Insights**:  
        These graphs help traders visualize how options respond to different market conditions, aiding strategy development and risk management by highlighting sensitivity to economic changes.

        6. **Practical Application**:  
        The comparison clarifies how the Black-Scholes model prices options under various scenarios, making theoretical dynamics accessible and actionable for market participants.
        """)
    
    elif st.session_state.page == "Compare Put option prices":
        st.title('Comparative Analysis of Put Option Price vs. Stock Price Graphs')
        col1, col2, col3 = st.columns(3)
        
        # Sliders for parameter set 1
        exerciseprice1 = col1.slider("Exercise Price 1", min_value=10, max_value=500, value=200)
        maturity1 = col1.slider("Maturity 1 (years)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        riskfreerate1 = col1.slider("Risk-Free Rate 1", min_value=0.0, max_value=0.2, step=0.01, value=0.05)
        sigma1 = col1.slider("Volatility 1", min_value=0.01, max_value=1.0, step=0.01, value=0.2)

        # Sliders for parameter set 2
        exerciseprice2 = col2.slider("Exercise Price 2", min_value=10, max_value=500, value=250)
        maturity2 = col2.slider("Maturity 2 (years)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        riskfreerate2 = col2.slider("Risk-Free Rate 2", min_value=0.0, max_value=0.2, step=0.01, value=0.05)
        sigma2 = col2.slider("Volatility 2", min_value=0.01, max_value=1.0, step=0.01, value=0.3)

        # Sliders for parameter set 3
        exerciseprice3 = col3.slider("Exercise Price 3", min_value=10, max_value=500, value=300)
        maturity3 = col3.slider("Maturity 3 (years)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)
        riskfreerate3 = col3.slider("Risk-Free Rate 3", min_value=0.0, max_value=0.2, step=0.01, value=0.05)
        sigma3 = col3.slider("Volatility 3", min_value=0.01, max_value=1.0, step=0.01, value=0.4)

        # Generate and display the dynamic graph
        fig = compare_put_option_vs_stock_price(exerciseprice1, maturity1, riskfreerate1, sigma1,
                                                exerciseprice2, maturity2, riskfreerate2, sigma2,
                                                exerciseprice3, maturity3, riskfreerate3, sigma3)
        st.pyplot(fig)
        st.markdown("""
        ### Key Insights from Comparing Put Option Price vs. Stock Price Graphs

        1. **Exercise Price Impact**:  
        Higher exercise prices increase the value of put options as they enhance the likelihood of being profitable. Comparing different exercise prices shows the sensitivity of options to strike price variations.

        2. **Maturity Effect**:  
        Longer maturities increase put option prices due to higher time value, reflecting more time for the stock to fall below the exercise price. This highlights the diminishing time value as expiration nears.

        3. **Risk-Free Rate Influence**:  
        A higher risk-free rate decreases put option prices since future payoffs are discounted more. Changes in this rate can significantly affect option values, especially in low-interest environments.

        4. **Volatility's Role**:  
        Increased volatility raises put option prices by enhancing the probability of profitable outcomes. Graphs with varying volatilities demonstrate the "insurance" value of options amid market uncertainty.

        5. **Market Insights**:  
        These graphs help traders visualize how options respond to different market conditions, aiding strategy development and risk management by highlighting sensitivity to economic changes.

        6. **Practical Application**:  
        The comparison clarifies how the Black-Scholes model prices options under various scenarios, making theoretical dynamics accessible and actionable for market participants.
        """)

        

    elif st.session_state.page == "Put-Call Parity":
        st.title('Put-Call Parity Analysis with Adjustable Parameter')
        st.sidebar.write("Adjust parameters for Put-Call Parity:")
        exerciseprice = st.sidebar.slider("Exercise Price", 10, 1000, 450)
        maturity = st.sidebar.slider("Maturity (in years)", 5, 10, 6, 1)
        riskfreerate = st.sidebar.slider("Risk-Free Rate", -0.01, 0.10, 0.04)
        sigma = st.sidebar.slider("Volatility", 0.1, 2.0, 0.4029, 0.01)
        
        st.write("### Call Option Price (Ct)")
        fig_ct = plot_call_option_price_vs_stock_price(exerciseprice, maturity, riskfreerate, sigma)
        st.pyplot(fig_ct)

        st.write("### Present Value of Exercise Price (X / (1+r)^T)")
        fig_xr = plot_present_value_of_exercise_price(exerciseprice, maturity, riskfreerate)
        st.pyplot(fig_xr)

        st.write("### Put-Call Parity (Combined)")
        fig_combined = plot_put_call_parity_graph(exerciseprice, maturity, riskfreerate, sigma)
        st.pyplot(fig_combined)

        st.markdown("""### Understanding Put-Call Parity

        #### 1. **Definition of Put-Call Parity**
        Put-Call Parity is a fundamental principle in options pricing that shows the relationship between the prices of European put and call options. It states that the value of a call option, combined with the present value of the exercise price, should be equal to the value of a put option and the current stock price. The equation is expressed as:

        \[ Ct + X / (1+r)^T = S + Pt \]

        Where:
        - \( Ct \) = Call option price
        - \( Pt \) = Put option price
        - \( S \) = Stock price
        - \( X \) = Exercise (strike) price
        - \( r \) = Risk-free rate
        - \( T \) = Time to maturity

        #### 2. **Call Option Price (Ct) vs Stock Price (S)**
        - **Graph Overview**: This graph illustrates the price of a call option (Ct) as the stock price (S) varies.
        - **Key Insights**:
        - The call option price increases as the stock price increases.
        - This behavior reflects the potential value of exercising the option if the stock price exceeds the exercise price.

        #### 3. **Present Value of Exercise Price (X / (1+r)^T)**
        - **Graph Overview**: This graph shows the present value of the exercise price, which is calculated by discounting the exercise price \( X \) by the risk-free rate \( r \) over the time to maturity \( T \).
        - **Key Insights**:
        - The present value of the exercise price remains constant across different stock prices.
        - A higher risk-free rate decreases the present value, making it cheaper to exercise the option in the future.

        #### 4. **Put-Call Parity (Combined Graph)**
        - **Graph Overview**: The combined graph shows the Left-Hand Side (LHS) and Right-Hand Side (RHS) of the Put-Call Parity equation as stock prices vary.
        - **LHS**: \( Ct + X / (1+r)^T \) - The sum of the call option price and the present value of the exercise price.
        - **RHS**: \( S + Pt \) - The sum of the stock price and the put option price.
        - **Key Insights**:
        - The Put-Call Parity relationship holds when the LHS and RHS lines overlap, demonstrating that both sides of the equation yield the same value.
        - Deviations from this parity can indicate potential arbitrage opportunities.

        #### 5. **Implications of Put-Call Parity**
        - **Arbitrage Opportunities**: If the Put-Call Parity does not hold, traders can exploit this by buying undervalued options and selling overvalued ones.
        - **Market Efficiency**: Put-Call Parity is often used to test the efficiency of the options market. If the parity relationship consistently holds, the market is likely efficient.
        - **Pricing Consistency**: This principle ensures consistent pricing between puts and calls, helping traders and investors to evaluate and manage risk effectively.

        By understanding Put-Call Parity, you gain insights into the interconnected nature of option pricing and the importance of market equilibrium in financial markets.
        """)
    
    elif st.session_state.page == "Intrinsic Payoff":
        st.title('Intrinsic Payoff vs. Stock Price')
        
        # Slider for exercise price
        exerciseprice = st.sidebar.slider("Exercise Price", min_value=10, max_value=500, value=200)
        
        # Generate and display the dynamic graph
        fig = intrinsic_payoff_vs_stock_price(exerciseprice)
        st.pyplot(fig)
        
        st.markdown("""
        ### Key Insights from the Intrinsic Payoff vs. Stock Price Graph

        1. **Intrinsic Value**:  
        The graph shows how the intrinsic value of the option changes with the stock price, given a fixed exercise price. The intrinsic value represents the payoff that the option holder would receive if the option were exercised immediately.

        2. **Exercise Price Impact**:  
        As the exercise price increases, the intrinsic value decreases, reflecting that the option becomes less valuable as it is further out-of-the-money.

        3. **Stock Price Sensitivity**:  
        When the stock price is below the exercise price, the intrinsic value is zero. When the stock price exceeds the exercise price, the intrinsic value increases linearly with the stock price, highlighting the direct relationship between the stock price and the option's payoff.

        4. **Practical Application**:  
        This graph helps traders and investors understand the immediate payoff of an option based on current stock prices, aiding in decisions related to option exercises and hedging strategies.
        """)
    elif st.session_state.page == "Binomial Option Pricing":
        st.title("Binomial Option Pricing with Hedged Portfolio")
        st.sidebar.header("Input Parameters")

        # Inputs
        S0 = st.sidebar.number_input("Current Stock Price (S₀)", value=100.0, min_value=0.0, step=1.0)
        U = st.sidebar.number_input("Up Factor (U)", value=1.2, min_value=1.0, step=0.1)
        d = st.sidebar.number_input("Down Factor (d)", value=0.9, min_value=0.0, max_value=1.0, step=0.1)
        X = st.sidebar.number_input("Exercise Price (X)", value=110.0, min_value=0.0, step=1.0)
        T = st.sidebar.number_input("Time to Expiration (T)", value=1, min_value=1, step=1)
        rf = st.sidebar.number_input("Risk-Free Rate (r%)", value=10.0, min_value=0.0, step=0.1)

        # Calculate option price and hedge ratio
        C, hedge_ratio, S_up, S_down, payoff_up, payoff_down = calculate_option_price(S0, U, d, X, T, rf)

        st.subheader("Option Pricing Results")
        st.write(f"**Call Option Price (C):** ${C:.2f}")
        st.write(f"**Hedge Ratio:** {hedge_ratio:.2f}")

        # Plot the binomial tree
        st.subheader("Binomial Price Tree")
        fig = binomial_option_pricing(S0, U, d, C, S_up, S_down, payoff_up, payoff_down)
        st.pyplot(fig)
    
if __name__ == "__main__":
    app()
