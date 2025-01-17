import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import yfinance as yf

def stocks_description():
    st.write("""
    ## Understanding stocks
    """)

def general_dividend_discount_model():
    st.write("""
    ## General Dividend Discount Model (GDDM)
    The General Dividend Discount Model (GDDM) determines the value of a stock based on the sum of all future dividend payments, plus the terminal stock price, discounted back to their present value.
    """)
    st.latex(r"P_0 = \sum \left( \frac{D_t}{(1 + r)^t} \right) + \frac{P_t}{(1 + r)^t}")

    st.write("""
    **Where:**
    - \\( P0 \\) is the current price of the stock.
    - \\( Dt \\) is the dividend expected at time \\( t \\).
    - \\( Pt \\) is the terminal stock price (future value).
    - \\( r \\) is the required rate of return.
    - \\( t \\) is the time period.
    """)

    # Fixed number of periods
    periods = 5

    # Input parameters
    discount_rate = st.number_input("Enter the required rate of return (r) as a percentage:", value=10.0, step=0.1) / 100
    terminal_price = st.number_input("Enter the terminal stock price (P_t):", value=100.0, step=1.0)

    # Input dividends for each period
    dividends = [st.number_input(f"Enter the dividend for year {t} (D_{t}):", value=1.0 + t * 0.2, step=0.1) for t in range(1, periods + 1)]

    # Calculating the present value of each dividend
    present_values_dividends = [dividends[t] / (1 + discount_rate) ** (t + 1) for t in range(periods)]
    
    # Calculating the present value of the terminal stock price
    present_value_terminal_price = terminal_price / (1 + discount_rate) ** periods

    # Calculating the total stock price (P₀)
    total_present_value = sum(present_values_dividends) + present_value_terminal_price
    st.write(f"**The calculated present value of the stock (P₀) is: ${total_present_value:.2f}**")

    # Plotting the dividends and their present values
    fig, ax1 = plt.subplots()

    # Primary y-axis for dividends
    ax1.plot(range(1, periods + 1), dividends, label='Dividends (Dₜ)', marker='o', color='blue')
    # Labeling each dividend point with D1, D2, etc.
    for i, dividend in enumerate(dividends):
        ax1.text(i + 1, dividend, f'D{i + 1} (${dividend:.2f})', ha='right', va='bottom', color='blue')

    # Adding the purple star for total stock price and labeling it
    ax1.scatter([0], [total_present_value], color='purple', label='Stock Price (P₀)', marker='*', s=200)
    ax1.text(0, total_present_value, f'Final P₀ (${total_present_value:.2f})', ha='left', va='center', color='purple')

    ax1.set_xlabel('Period')
    ax1.set_ylabel('Stock Price ($)')
    
    # Secondary y-axis for stock price
    ax2 = ax1.twinx()
    ax2.scatter([0] * periods, present_values_dividends, label='Present Value of Dividends', color='orange', marker='x')
    
    # Labeling each present value with D1, D2, etc.
    for i, pv_dividend in enumerate(present_values_dividends):
        ax2.text(0, pv_dividend, f'D{i + 1} (${pv_dividend:.2f})', ha='right', va='bottom', color='orange')
    
    

    # Titles and Legends
    ax1.set_title('Dividends, Present Values, and Stock Price')
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    
    # Display the graph
    st.pyplot(fig)

    # Example calculation details
    st.write('''
### Example Calculation

Suppose the required rate of return (r) is 10%, the dividends for each period \\( Dt \\) are \\( [1.2, 1.4, 1.6, 1.8, 2.0] \\), and the terminal price (Pt) at the end of 5 years is $100. 

#### Calculation Details

- **Year 1**: 
  $$ \\frac{1.2}{(1 + 0.1)^1} \\approx 1.09 $$

- **Year 2**: 
  $$ \\frac{1.4}{(1.1)^2} \\approx 1.16 $$

- **Year 3**: 
  $$ \\frac{1.6}{(1.1)^3} \\approx 1.20 $$

- **Year 4**: 
  $$ \\frac{1.8}{(1.1)^4} \\approx 1.23 $$

- **Year 5**: 
  $$ \\frac{2.0}{(1.1)^5} \\approx 1.24 $$

Terminal stock price:

$$ \\frac{100}{(1.1)^5} \\approx 62.09 $$

Adding these values gives:
$$
P_0 = 1.09 + 1.16 + 1.20 + 1.23 + 1.24 + 62.09 \\approx 68.01
$$

This final value, **$68.01**, is represented as the purple star on the graph.
''')
    
def constant_dividend_growth_model():
    st.write("""
    ## Constant Dividend Growth Model (CDGM)
    The Constant Dividend Growth Model (CDGM) estimates the value of a stock based on the expected dividends growing at a constant rate forever.
    
    **Formula:**
    
    P₀ =   D₀ * (1 + g) / (r - g)
    
    **Where:**
    
    - P₀ is the current price of the stock.
    - D₀ is the current dividend (dividend for the current year).
    - r is the required rate of return.
    - g is the constant growth rate of dividends.
    """)

    # Input parameters
    current_dividend = st.number_input("Enter the current dividend (D₀):", value=1.0, step=0.1)
    required_rate_of_return = st.number_input("Enter the required rate of return (r) as a percentage:", value=10.0, step=0.1) / 100
    dividend_growth_rate = st.number_input("Enter the constant dividend growth rate (g) as a percentage:", value=2.0, step=0.1) / 100

    # Ensure that the growth rate is less than the required rate of return
    if dividend_growth_rate >= required_rate_of_return:
        st.error("The growth rate (g) must be less than the required rate of return (r). Please adjust the values.")
    else:
        # Step-by-step calculation
        st.write(f"### Step-by-Step Calculation:")
        
        st.write(f"1. **Input Values:**")
        st.write(f"   - Required Rate of Return (r): {required_rate_of_return * 100:.2f}%")
        st.write(f"   - Dividend Growth Rate (g): {dividend_growth_rate * 100:.2f}%")
        st.write(f"   - Current Year's Dividend (D₀): ${current_dividend:.2f}")

        # Step 2: Calculate the stock price
        stock_price = (current_dividend * (1 + dividend_growth_rate)) / (required_rate_of_return - dividend_growth_rate)
        st.write(f"3. **Calculate the Stock Price (P₀):**")
        st.write(f"   - P₀ = {current_dividend:.2f} * (1 + {dividend_growth_rate:.4f}) / ({required_rate_of_return:.4f} - {dividend_growth_rate:.4f}) = **${stock_price:.2f}**")

        st.write(f"**The intrinsic value of the stock is: ${stock_price:.2f}**")

        # Plotting Stock Value vs Dividend Growth Rate (g)
        growth_rates = np.linspace(0, required_rate_of_return, 50)
        values_vs_g = [(current_dividend * (1 + g)) / (required_rate_of_return - g) for g in growth_rates]


        fig, ax = plt.subplots()
        ax.plot(growth_rates * 100, values_vs_g, label="Stock Value vs Growth Rate (g)", color="blue")
        ax.axvline(required_rate_of_return * 100, color='green', linestyle=':', label='Required Rate (r)')
        ax.scatter(dividend_growth_rate * 100, stock_price, color='red', marker='o', label=f'Current Growth Rate (g = {dividend_growth_rate * 100:.2f}%)')
        ax.set_xlabel("Dividend Growth Rate (g) %")
        ax.set_ylabel("Stock Value ($)")
        ax.set_title("Stock Value vs Dividend Growth Rate (g)")
        ax.legend()
        st.pyplot(fig)

        # Plotting Stock Value vs Required Rate of Return (r)
        required_rates = np.linspace(dividend_growth_rate + 0.01, required_rate_of_return + 0.05, 50)
        values_vs_r = [(current_dividend * (1 + dividend_growth_rate)) / (r - dividend_growth_rate) for r in required_rates]

        fig, ax = plt.subplots()
        ax.plot(required_rates * 100, values_vs_r, label="Stock Value vs Required Rate of Return (r)", color="red")
        ax.axvline(dividend_growth_rate * 100, color='orange', linestyle=':', label='Dividend Growth Rate (g)')
        ax.scatter(required_rate_of_return * 100, stock_price, color='blue', marker='o', label=f'Current Required Rate (r = {required_rate_of_return * 100:.2f}%)')
        ax.set_xlabel("Required Rate of Return (r) %")
        ax.set_ylabel("Stock Value ($)")
        ax.set_title("Stock Value vs Required Rate of Return (r)")
        ax.legend()
        st.pyplot(fig)

        st.write(''' 
        - **Stock Price vs Dividend Growth Rate (g)**: 
          This graph demonstrates how the stock price (P₀) changes as the dividend growth rate (`g`) varies. The required rate of return (`r`) is held constant.
          As `g` increases, the stock price increases, reflecting the higher expected future dividends.

        - **Stock Price vs Required Rate of Return (r)**:
          This graph shows how the stock price (P₀) is affected by changes in the required rate of return (`r`). The growth rate (`g`) is held constant.
          As `r` increases, the stock price decreases, indicating that a higher required return reduces the present value of expected future dividends.
        ''')

def two_stage_growth_model():
    st.write("""
    ## Two-Stage Dividend Growth Model (TSDGM)
    The Two-Stage Dividend Growth Model accounts for two phases of dividend growth: 
    a high-growth phase followed by a perpetual constant growth phase.
    """)

    # Input parameters
    current_dividend = st.number_input("Enter the current dividend (D₀):", value=1.0, step=0.1)
    high_growth_rate = st.number_input("Enter the initial high growth rate (g₁) as a percentage:", value=15.0, step=0.1) / 100
    perpetual_growth_rate = st.number_input("Enter the perpetual growth rate (g₂) as a percentage:", value=5.0, step=0.1) / 100
    required_rate_of_return = st.number_input("Enter the required rate of return (r) as a percentage:", value=20.0, step=0.1) / 100
    high_growth_years = st.number_input("Enter the number of years of high growth (T):", value=3, step=1)

    # Error handling: Ensure growth rates are valid
    if perpetual_growth_rate >= required_rate_of_return:
        st.error("The perpetual growth rate (g₂) must be less than the required rate of return (r). Please adjust the values.")
        return

    # Calculate present value of dividends during high-growth phase
    dividends_high_growth = [
        current_dividend * (1 + high_growth_rate) ** t for t in range(1, high_growth_years + 1)
    ]
    present_value_high_growth = [
        dividend / (1 + required_rate_of_return) ** t for t, dividend in enumerate(dividends_high_growth, start=1)
    ]

    # Calculate terminal value after high-growth phase (perpetual growth phase)
    terminal_dividend = current_dividend * (1 + high_growth_rate) ** high_growth_years
    terminal_value = (terminal_dividend * (1 + perpetual_growth_rate)) / (required_rate_of_return - perpetual_growth_rate)
    present_value_terminal_value = terminal_value / (1 + required_rate_of_return) ** high_growth_years

    # Calculate total present value (P₀)
    stock_price = sum(present_value_high_growth) + present_value_terminal_value

    st.write(f"**The calculated present value of the stock (P₀) is: ${stock_price:.2f}**")

    # Visualization with two y-axes
    periods = np.arange(1, high_growth_years + 2)
    total_dividends = dividends_high_growth + [terminal_dividend * (1 + perpetual_growth_rate)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot High Growth Phase Dividends (Blue Line)
    ax1.plot(periods[:-1], dividends_high_growth, label='High-Growth Phase Dividends', marker='o', color='blue', linestyle='--')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Dividends (scaled)', color='blue')

    # Set y-ticks for scaling to close values and starting from 1.0
    dividend_max = max(dividends_high_growth) + 1  # Ensure ticks accommodate maximum dividend
    ax1.set_yticks(np.arange(1.0, dividend_max + 1, 0.25))  # Adjusted to start from 1.0
    ax1.set_ylim(bottom=1.0)  # Set the lower limit to 1.0

    # Annotate dividend values on the left y-axis
    for t, dividend in enumerate(dividends_high_growth, start=1):
        ax1.text(t, dividend + 0.1, f"${dividend:.2f}", color='blue', ha='center')

    # Create a second y-axis for the stock price
    ax2 = ax1.twinx()
    ax2.set_ylabel('Stock Price ($)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Plot the stock price (final price) as a red star
    ax2.scatter([0], [stock_price], color='red', label='Final Stock Price (P₀)', marker='*', s=200)

    # Annotate the final stock price below the red star
    ax2.text(0, stock_price - 0.5, f' ${stock_price:.2f}', verticalalignment='top', horizontalalignment='center', color='red', fontsize=12)

    # Plot Perpetual Growth Phase (Green Line)
    ax1.plot(periods[-2:], [dividends_high_growth[-1], terminal_dividend * (1 + perpetual_growth_rate)], 
             label='Perpetual Growth Dividends', marker='*', color='green', linestyle='-')

    # Shaded Area for Present Value
    ax1.fill_between(periods, 0, total_dividends, color='lightblue', alpha=0.3, label='Dividend Growth Trajectory')

    # Set right y-axis ticks with gaps of 5
    ax2.set_yticks(np.arange(0, int(stock_price) + 6, 5))  # Adjusted to 5 intervals

    # Adding titles and legends
    ax1.set_title('Dividends Growth and Final Stock Price (Two-Stage Growth Model)')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')

    # Display the line chart
    st.pyplot(fig)


    # Understanding the Two-Stage Growth Model
    st.write(f"""
    ### Understanding the Model
    The 2-stage dividend growth model (also known as the multi-stage dividend discount model) calculates the price of a stock by considering different growth rates for different periods. 
    Here, we have two growth stages:

    - **Dividends grow at {high_growth_rate * 100:.2f}% for the first {high_growth_years} years.**
    - **After the third year, dividends grow at {perpetual_growth_rate * 100:.2f}% indefinitely.**

    The formula for the price of a stock in a multi-stage dividend growth model is:

    """)

    # LaTeX formula for the price of a stock
    st.latex(r'''
    P_0 = \sum_{t=1}^{T} \frac{D_t}{(1+r)^t} + \frac{P_T}{(1+r)^T}
    ''')

    st.write(f"""
    Where:
    - \(Dt\) = Dividend at time \(t\)
    - \(r\) = Required rate of return (20% or 0.20)
    - \(T\) = Number of periods with a high growth rate ({high_growth_years})
    - \(PT\) = Terminal price at the end of stage 1 (present value of all future dividends starting from year \(T+1\))

    ### Step-by-Step Solution:

    1. **Calculate the dividends for the first {high_growth_years} years:**
    """)

    # Step 1: Calculate and display dividends for the high growth phase
    for t in range(1, high_growth_years + 1):
        dividend = current_dividend * (1 + high_growth_rate) ** t
        st.write(f"- Dividend at year {t}: $D_{t} = D_0 \u00d7 (1 + g_1)^{t} = {current_dividend:.2f} \u00d7 (1 + {high_growth_rate * 100:.2f})^{t} = ${dividend:.2f}")

    st.write("")

    # Step 2: Calculate the terminal price at the end of year T
    terminal_dividend = current_dividend * (1 + high_growth_rate) ** high_growth_years
    st.write(f"2. **Calculate the terminal price $P_T$ at the end of year {high_growth_years}:**")
 
   # First line: Dividend calculation
    st.write(f"   - Dividend at year  {high_growth_years + 1}: D{high_growth_years + 1} = D{high_growth_years} \u00d7 (1 + g2) = {terminal_dividend:.2f} \u00d7 (1 + {perpetual_growth_rate * 100:.2f}) = {terminal_dividend * (1 + perpetual_growth_rate):.2f}")

    # Terminal price calculation
    P_T = (terminal_dividend * (1 + perpetual_growth_rate)) / (required_rate_of_return - perpetual_growth_rate)

    # Using double curly braces {{ and }} to escape curly braces in LaTeX
    st.latex(f""" P_{{{high_growth_years}}} = \\frac{{D_{{{high_growth_years + 1}}}}}{{r - g_2}} = \\frac{{{terminal_dividend * (1 + perpetual_growth_rate):.2f}}}{{{required_rate_of_return:.2f} - {perpetual_growth_rate:.2f}}} = {P_T:.2f} """)

    st.write("")

    # Step 3: Present value of dividends for the high-growth phase
    st.write(f"3. **Calculate the present value of dividends for the first {high_growth_years} years:**")
    for t in range(1, high_growth_years + 1):
        PV_D_t = dividends_high_growth[t - 1] / (1 + required_rate_of_return) ** t
        st.write(f"- Present Value of $D_{t}: \\frac{{D_{t}}}{{(1 + r)^{t}}} = \\frac{{{dividends_high_growth[t - 1]:.2f}}}{{(1 + {required_rate_of_return:.2f})^{t}}} = ${PV_D_t:.2f}")

    st.write("")

    # Step 4: Present value of terminal price
    st.write(r"4. **Calculate the present value of the terminal price \(PT\):**")
    PV_P_T = P_T / (1 + required_rate_of_return) ** high_growth_years

# Use st.latex for rendering LaTeX directly
    st.write("- Present Value of Terminal Price:")
    st.latex(f"    \\frac{{P_T}}{{(1 + r)^T}} = \\frac{{{P_T:.2f}}}{{(1 + {required_rate_of_return:.2f})^{{{high_growth_years}}}}} = {PV_P_T:.2f}")

    st.write("")

    # Final Calculation
   # Assuming present_value_high_growth and PV_P_T are already calculated
    total_PV = sum(present_value_high_growth) + PV_P_T

    # Using st.write to show a plain text explanation and st.latex to render the formula
    st.latex(f"Total Present Value (P₀) = sum of PV(D₁) + sum of PV(D₂) + ... + PV(D_{{{high_growth_years}}}) + PV(P_T)")
    st.write(f"**= ${total_PV:.2f}**")

def FCFF(fcff_list):
    st.write("""
    ## Free Cash Flow (FCFF) Model (Adjusted for Given FCFF Values)
    This model calculates the value of a firm based on Free Cash Flows (FCFF) over a fixed high-growth phase followed by a stable-growth phase.
    """)

    # Input parameters
    tax_rate = st.number_input("Enter Tax Rate (%):", value=36.0, step=1.0) / 100
    risk_free_rate = st.number_input("Enter Risk Free Rate (%):", value=7.5, step=0.1) / 100
    market_risk_premium = st.number_input("Enter Market Risk Premium (%):", value=5.5, step=0.1) / 100

    # Inputs for high-growth phase (fixed to 5 years)
    high_growth_years = 5  # Fixed to 5 years
    beta_high = st.number_input("Beta for high-growth phase:", value=1.25, step=0.01)
    cost_of_debt_high = st.number_input("Cost of Debt (high-growth phase, %):", value=9.5, step=0.1) / 100
    
    # Inputs for stable-growth phase
    stable_growth_rate = st.number_input("Stable growth rate (%):", value=5.0, step=0.1) / 100
    beta_stable = st.number_input("Beta for stable-growth phase:", value=1.0, step=0.01)
    cost_of_debt_stable = st.number_input("Cost of Debt (stable phase, %):", value=8.5, step=0.1) / 100
    
     # Inputs for D/E ratios
    debt_to_equity_high = st.number_input("Debt-to-Equity Ratio (High-Growth Phase):", value=0.35, step=0.01)
    debt_to_equity_stable = st.number_input("Debt-to-Equity Ratio (Stable-Growth Phase):", value=0.25, step=0.01)

    # Step 1: Cost of Equity Calculation (using CAPM: Cost of Equity = Risk-Free Rate + Beta * Market Risk Premium)
    cost_of_equity_high = risk_free_rate + beta_high * market_risk_premium
    cost_of_equity_stable = risk_free_rate + beta_stable * market_risk_premium
    st.write(f"**Cost of Equity in High-Growth Phase:** {cost_of_equity_high * 100:.2f}%")
    st.write(f"**Cost of Equity in Stable-Growth Phase:** {cost_of_equity_stable * 100:.2f}%")

    # Calculate We and Wd based on D/E ratio
    def calculate_weights(debt_to_equity_ratio):
        total_value_ratio = 1 + debt_to_equity_ratio  # V/E = 1 + D/E
        weight_equity = 1 / total_value_ratio        # E/V = 1 / (1 + D/E)
        weight_debt = debt_to_equity_ratio / total_value_ratio  # D/V = (D/E) / (1 + D/E)
        return weight_equity, weight_debt

    # Calculate weights for high-growth phase
    we_high, wd_high = calculate_weights(debt_to_equity_high)

    # Calculate weights for stable-growth phase
    we_stable, wd_stable = calculate_weights(debt_to_equity_stable)

    # Calculate WACC for high-growth phase
    wacc_high = we_high * cost_of_equity_high + wd_high * cost_of_debt_high * (1 - tax_rate)

    # Calculate WACC for stable-growth phase
    wacc_stable = we_stable * cost_of_equity_stable + wd_stable * cost_of_debt_stable * (1 - tax_rate)

    # Display WACC results
    st.write(f"**WACC for High-Growth Phase:** {wacc_high * 100:.2f}%")
    st.write(f"**WACC for Stable-Growth Phase:** {wacc_stable * 100:.2f}%")
        
    
    # Step 3: Use Provided FCFF Values for High-Growth Phase (Skip recalculating FCFF)
    st.write("### FCFF Values for High-Growth Phase (Provided):")
    for year, fcff in enumerate(fcff_list, start=1):
        st.write(f"Year {year}:")
        st.write(f"- **FCFF Value:** {fcff:.2f}")

   
    # # FCFF in the terminal year
    # fcff_terminal = (ebit_terminal - tax_rate * ebit_terminal) + depreciation_terminal - capital_expenditures_terminal - delta_working_capital_terminal
    # st.write("### Terminal Year FCFF Calculation:")
    # st.write(f"- **Terminal EBIT (after stable growth rate):** {ebit_terminal:.2f} ")
    # st.write(f"- **FCFF for Terminal Year:**")
    # st.write(f"  = ({ebit_terminal:.2f} - {tax_rate:.2f} * {ebit_terminal:.2f}) + {depreciation_terminal:.2f} - {capital_expenditures_terminal:.2f} - {delta_working_capital_terminal:.2f}")
    # st.write(f"  = {fcff_terminal:.2f}")

    # # Terminal Value calculation using the perpetuity formula
    # terminal_value = fcff_terminal / (wacc_stable - stable_growth_rate)
    # st.write(f"- **Terminal Value of the Firm:**")
    # st.write(f" = FCFF terminal / (WACC stable - stable growth rate)")
    # st.write(f" = {fcff_terminal:.2f}/({wacc_stable} - {stable_growth_rate})")
    # st.write(f"**Terminal Value (TV):** {terminal_value:.2f} ")

    # # Step 5: Calculating Present Value of Terminal Value
    # pv_terminal_value = terminal_value / ((1 + wacc_high) ** high_growth_years)
    # st.write("### Present Value of Terminal Value:")
    # st.write(f" = Terminal Value / (1 + WACC high) ^ High growth years")
    # st.write(f" = {terminal_value:.2f}/(1+ {wacc_high:.2f} ^ {high_growth_years})")
    # st.write(f" **Present Value of Terminal Value:** {pv_terminal_value:.2f} ")

    fcff_last_high = fcff_list[-1]
    # Terminal Value calculation using the perpetuity formula
    # FCFFt+1 = FCFFt * (1 + stable growth rate)
    fcff_t1 = fcff_last_high * (1 + stable_growth_rate)
    terminal_value = fcff_t1 / (wacc_stable - stable_growth_rate)

    # Display FCFF_t+1 and Terminal Value
    st.write("### Terminal Value Calculation:")
    st.write(f"- **FCFF in the first stable year (FCFFt+1):**")
    st.write(f" = FCFF at the end of high-growth phase × (1 + Stable growth rate)")
    st.write(f" = {fcff_last_high:.2f} × (1 + {stable_growth_rate:.2f})")
    st.write(f" = {fcff_t1:.2f}")
    st.write(f"- **Terminal Value (TV):**")
    st.write(f" = FCFFt+1 / (WACC stable - Stable growth rate)")
    st.write(f" = {fcff_t1:.2f} / ({wacc_stable:.2f} - {stable_growth_rate:.2f})")
    st.write(f"**Terminal Value (TV):** {terminal_value:.2f}")

    # Present Value of Terminal Value
    pv_terminal_value = terminal_value / ((1 + wacc_high) ** high_growth_years)

    # Display Present Value of Terminal Value
    st.write("### Present Value of Terminal Value:")
    st.write(f" = Terminal Value / (1 + WACC high) ^ High growth years")
    st.write(f" = {terminal_value:.2f} / ((1 + {wacc_high:.2f}) ^ {high_growth_years})")
    st.write(f"**Present Value of Terminal Value:** {pv_terminal_value:.2f}")


    # Step 6: Calculating Present Value of FCFFs in the high-growth phase
    pv_fcff_high_growth = sum(fcff / ((1 + wacc_high) ** year) for year, fcff in enumerate(fcff_list, start=1))
    st.write("### Present Value of FCFFs during High-Growth Phase:")
    for year, fcff in enumerate(fcff_list, start=1):
        st.write(f"Year {year}:")
        st.write(f"- **FCFF Year {year}:** {fcff:.2f}")
        st.write(f"- **PV of FCFF Year {year}** = {fcff:.2f} / (1 + {wacc_high:.2f})^{year} = {fcff / ((1 + wacc_high) ** year):.2f}")
    st.write(f"**Total PV of FCFFs in High-Growth Phase:** {pv_fcff_high_growth:.2f} ")

    # Step 7: Calculating the Total Firm Value
    firm_value = pv_fcff_high_growth + pv_terminal_value
    st.write("### Total Firm Value:")
    st.write(f" = Present value of FCFF + Present value of Terminal value")
    st.write(f" = {pv_fcff_high_growth:.2f} + {pv_terminal_value:.2f}")
    st.write(f"**Total Value of the Firm:** {firm_value:.2f}")



    # Plot FCFF for high-growth phase and terminal FCFF
    years = list(range(1, high_growth_years + 2))

    # Add FCFF for the terminal year (Year 6)
    fcff_values = fcff_list + [fcff_t1]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(years, fcff_values, marker='o', label="FCFF (High-Growth Phase + Terminal Year)", color="b")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("FCFF ($)", color="b")
    ax1.tick_params(axis='y', labelcolor="b")
    ax1.grid(True)

    # Create a second y-axis for Total Firm Value
    ax2 = ax1.twinx()
    ax2.plot([0], [firm_value], marker="*", markersize=10, color="r", label="Total Firm Value", linestyle='None')
    ax2.set_ylabel("Firm Value ($)", color="g")
    ax2.tick_params(axis='y', labelcolor="g")

    
    # Annotations
    for i, fcff in enumerate(fcff_values):
        ax1.annotate(f"{fcff:.2f}", (years[i], fcff), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)
    ax2.annotate(f"{firm_value:.2f}",
                 xy=(0, firm_value), 
                 xycoords='data',
                 xytext=(10, 0),
                 textcoords='offset points',
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=12, color="r")

    # Title and legends
    ax1.set_title("FCFF Valuation Model with Total Firm Value")
    ax1.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
    ax2.legend(loc='upper left',bbox_to_anchor=(1.05, 0.925))

    st.pyplot(fig)

def free_cash_flows():
    st.write("""
    ## Free Cash Flow to Firm (FCFF) Model
    This model calculates the value of a firm based on Free Cash Flows (FCFF), broken down into a high-growth phase followed by a stable-growth phase.
    ### Explanation of Key Variables
    - **EBIT**: Earnings Before Interest and Taxes, a measure of a company's profitability.
    - **Capital Expenditures**: Funds used by the firm to acquire or maintain physical assets.
    - **Depreciation**: Reduction in asset value over time, added back as it doesn’t involve cash outflows.
    - **Working Capital**: Funds required for day-to-day operations, calculated here as a percentage of revenues.
    """)

    # Input parameters for the firm in 1994
    ebit = st.number_input("Enter EBIT ($ millions):", value=532.0, step=1.0)
    capital_expenditures = st.number_input("Enter Capital Expenditures:", value=310.0, step=1.0)
    depreciation = st.number_input("Enter Depreciation:", value=207.0, step=1.0)
    revenues = st.number_input("Enter Revenues:", value=7230.0, step=1.0)
    current_working_capital = st.number_input("Enter current Working capital:", value=1952.0, step=1.0)
    previous_working_capital = st.number_input("Enter previous Working capital:", value=1818.1, step=1.0)
    tax_rate = st.number_input("Enter Tax Rate (%):", value=36.0, step=1.0) / 100
    risk_free_rate = st.number_input("Enter Risk Free Rate (%):", value=7.5, step=0.1) / 100
    market_risk_premium = st.number_input("Enter Market Risk Premium (%):", value=5.5, step=0.1) / 100

    # Inputs for high-growth phase
    high_growth_years = st.number_input("High growth phase (years):", value=5, step=1)
    ebit_growth_rate = st.number_input("EBIT growth rate in high-growth phase (%):", value=8.0, step=0.1) / 100
    beta_high = st.number_input("Beta for high-growth phase:", value=1.25, step=0.01)
    cost_of_debt_high = st.number_input("Cost of Debt (high-growth phase, %):", value=9.5, step=0.1) / 100
    debt_ratio_high = st.number_input("Debt Ratio for high-growth phase (%):", value=50.0, step=1.0) / 100

    # Inputs for stable-growth phase
    stable_growth_rate = st.number_input("Stable growth rate (%):", value=5.0, step=0.1) / 100
    beta_stable = st.number_input("Beta for stable-growth phase:", value=1.0, step=0.01)
    cost_of_debt_stable = st.number_input("Cost of Debt (stable phase, %):", value=8.5, step=0.1) / 100
    debt_ratio_stable = st.number_input("Debt Ratio for stable-growth phase (%):", value=25.0, step=1.0) / 100

    # Step 1: Cost of Equity Calculation (using CAPM: Cost of Equity = Risk-Free Rate + Beta * Market Risk Premium)
    cost_of_equity_high = risk_free_rate + beta_high * market_risk_premium
    cost_of_equity_stable = risk_free_rate + beta_stable * market_risk_premium
    st.write(f"**Cost of Equity in High-Growth Phase:** {cost_of_equity_high * 100:.2f}%")
    st.write(f"**Cost of Equity in Stable-Growth Phase:** {cost_of_equity_stable * 100:.2f}%")

    # Step 2: WACC (Weighted Average Cost of Capital)
    # WACC formula: WACC = (Equity Proportion * Cost of Equity) + (Debt Proportion * Cost of Debt * (1 - Tax Rate))
    wacc_high = (1 - debt_ratio_high) * cost_of_equity_high + debt_ratio_high * cost_of_debt_high * (1 - tax_rate)
    wacc_stable = (1 - debt_ratio_stable) * cost_of_equity_stable + debt_ratio_stable * cost_of_debt_stable * (1 - tax_rate)
    st.write(f"**WACC for High-Growth Phase:** {wacc_high * 100:.2f}%")
    st.write(f"**WACC for Stable-Growth Phase:** {wacc_stable * 100:.2f}%")

    # Calculate delta working capital
    delta_working_capital = current_working_capital - previous_working_capital
    st.write(f"**Change in Working Capital (Δ Working Capital):** {delta_working_capital:.2f} ")

    # Step 3: Calculating FCFF for each year in the high-growth phase
    fcff_list = []
    st.write("### Free Cash Flow to Firm (FCFF) Calculations for High-Growth Phase:")
    for year in range(1, high_growth_years + 1):
        ebit_year = ebit * (1 + ebit_growth_rate) ** year
        depreciation_year = depreciation * (1 + ebit_growth_rate) ** year
        capital_expenditures_year = capital_expenditures * (1 + ebit_growth_rate) ** year
        delta_working_capital_year = delta_working_capital * (1 + ebit_growth_rate) ** year
        fcff = (ebit_year - tax_rate * ebit_year) + depreciation_year - capital_expenditures_year - delta_working_capital_year
        fcff_list.append(fcff)
        st.write(f"Year {year}:")
        st.write(f"- **EBIT (after growth rate):** {ebit_year:.2f}")
        st.write(f"- **FCFF Calculation**:")
        st.write(f"  = ({ebit_year:.2f} - {tax_rate:.2f} * {ebit_year:.2f}) + {depreciation_year:.2f} - {capital_expenditures_year:.2f} - {delta_working_capital_year:.2f}")
        st.write(f"  = {fcff:.2f}")

    # Step 4: Calculate Terminal Value at the end of the high-growth phase
    # EBIT in the final year of the high-growth phase
    ebit_terminal = ebit * (1 + ebit_growth_rate) ** high_growth_years * (1 + stable_growth_rate)
    
    # In the terminal phase, capital expenditures equal depreciation
    capital_expenditures_terminal = depreciation_terminal = depreciation * (1 + ebit_growth_rate) ** high_growth_years * (1 + stable_growth_rate)

    # Working capital needs based on revenue growth
    delta_working_capital_terminal = delta_working_capital * (1 + ebit_growth_rate) ** high_growth_years * (1 + stable_growth_rate)

    # FCFF in the terminal year
    fcff_terminal = (ebit_terminal - tax_rate * ebit_terminal)  - delta_working_capital_terminal
    st.write("### Terminal Year FCFF Calculation:")
    st.write(f"- **Terminal EBIT (after stable growth rate):** {ebit_terminal:.2f} ")
    st.write(f"- **FCFF for Terminal Year:**")
    st.write(f" For Terminal EBIT, (capital expenditure-depreciation) values to 0")
    st.write(f"  = ({ebit_terminal:.2f} - {tax_rate:.2f} * {ebit_terminal:.2f}) - 0 -{delta_working_capital_terminal:.2f}")
    st.write(f"  = {fcff_terminal:.2f}")

    # Terminal Value calculation using the perpetuity formula
    terminal_value = fcff_terminal / (wacc_stable - stable_growth_rate)
    st.write(f"- **Terminal Value of the Firm:**")
    st.write(f" = FCFF terminal / (WACC stable - stable growth rate)")
    st.write(f" = {fcff_terminal:.2f}/({wacc_stable} - {stable_growth_rate})")
    st.write(f"**Terminal Value (TV):** {terminal_value:.2f} ")

    # Step 5: Calculating Present Value of Terminal Value
    pv_terminal_value = terminal_value / ((1 + wacc_high) ** high_growth_years)
    st.write("### Present Value of Terminal Value:")
    st.write(f" = Terminal Value / (1 + WACC high) ^ High growth years")
    st.write(f" = {terminal_value:.2f}/(1+ {wacc_high:.2f} ^ {high_growth_years})")
    st.write(f" **Present Value of Terminal Value:** {pv_terminal_value:.2f} ")

   
    # Step 6: Calculating Present Value of FCFFs in the high-growth phase
    pv_fcff_high_growth = sum(fcff / ((1 + wacc_high) ** year) for year, fcff in enumerate(fcff_list, start=1))
    st.write("### Present Value of FCFFs during High-Growth Phase:")
    for year, fcff in enumerate(fcff_list, start=1):
        st.write(f"Year {year}:")
        st.write(f"- **FCFF Year {year}:** {fcff:.2f}")
        st.write(f"- **PV of FCFF Year {year}** = {fcff:.2f} / (1 + {wacc_high:.2f})^{year} = {fcff / ((1 + wacc_high) ** year):.2f}")
    st.write(f"**Total PV of FCFFs in High-Growth Phase:** {pv_fcff_high_growth:.2f} ")

    # Step 7: Calculating the Total Firm Value
    firm_value = pv_fcff_high_growth + pv_terminal_value
    st.write("### Total Firm Value:")
    st.write(f" = Present value of FCFF + Present value of Terminal value")
    st.write(f" = {pv_fcff_high_growth:.2f} + {pv_terminal_value:.2f}")
    st.write(f"**Total Value of the Firm:** {firm_value:.2f}")

    # Plot FCFF for high-growth phase and terminal FCFF
    years = list(range(1, high_growth_years + 2))
    fcff_values = fcff_list + [fcff_terminal]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(years, fcff_values, marker='o', label="FCFF (High-Growth Phase + Terminal Year)", color="b")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("FCFF ($)", color="b")
    ax1.tick_params(axis='y', labelcolor="b")
    ax1.grid(True)

    # Create a second y-axis for Total Firm Value
    ax2 = ax1.twinx()
    ax2.plot([0], [firm_value], marker="*", markersize=10, color="r", label="Total Firm Value", linestyle='None')
    ax2.set_ylabel("Firm Value ($)", color="g")
    ax2.tick_params(axis='y', labelcolor="g")

    # Annotations
    for i, fcff in enumerate(fcff_values):
        ax1.annotate(f"{fcff:.2f}", (years[i], fcff), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=10)
    ax2.annotate(f"{firm_value:.2f}",
                 xy=(0, firm_value), 
                 xycoords='data',
                 xytext=(10, 0),
                 textcoords='offset points',
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=12, color="r")

    # Title and legends
    ax1.set_title("FCFF Valuation Model with Total Firm Value")
    ax1.legend(loc='upper left',bbox_to_anchor=(1.1, 1))
    ax2.legend(loc='upper left',bbox_to_anchor=(1.1, 0.925))

    st.pyplot(fig)

    # Load Excel file for download
    file_path = 'datasets/FCFF.xls'  # Replace with the path to your local Excel file
    df = pd.read_excel(file_path)

    # Convert to binary to make it downloadable
    with open(file_path, 'rb') as file:
        file_data = file.read()

    st.write("Download the Excel file below, which contains a solved example of real-time cash flows. The cells are pre-configured with formulas, allowing you to experiment with your own values. **Note:** Please refrain from changing the 'Years', as it is fixed at 8 years.")
# Download button for the Excel file
    st.download_button(
        label="Download Excel file",
        data=file_data,
        file_name="FCFF.xls",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def FCFF_webscraping():
    st.title("FCFF Calculation with Dynamic Inputs")
    
    # User inputs for ticker and date range
    st.sidebar.header("Company Details")
    ticker = st.sidebar.text_input("Enter Company Ticker:", "AAPL")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
     
    # Fetch financial data
    if st.sidebar.button("Fetch Financial Data"):
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow_statement = stock.cashflow
            
            st.write(f"### Income Statement for {ticker}")
            st.dataframe(income_statement)
            
            st.write(f"### Balance Sheet for {ticker}")
            st.dataframe(balance_sheet)
            
            st.write(f"### Cashflow Statement for {ticker}")
            st.dataframe(cashflow_statement)
            
        except Exception as e:
            st.error("Error fetching data. Please check the ticker or try again.")
    
    # FCFF Calculation
    st.header("Free Cash Flow to Firm (FCFF) Calculation")
    ebit = st.number_input("Enter EBIT:", step=1.0)
    cap_exp = st.number_input("Enter Capital Expenditures:", step=1.0)
    depreciation = st.number_input("Enter Depreciation:", step=1.0)
    change_in_nwc = st.number_input("Enter Change in Net Working Capital:", step=1.0)
    tax_rate = st.number_input("Enter Tax Rate (%):", step=0.1) / 100

    # Calculation fields for verification
    st.subheader("Enter calculated values for verification")
    t_ebit = st.number_input("T * EBIT:", step=1.0)
    cap_exp_minus_depr = st.number_input("(Cap Exp – Depreciation):", step=1.0)
    fcff = st.number_input("Free Cash Flow to Firm (FCFF):", step=1.0)

    # Perform calculations
    if st.button("Calculate and Verify"):
        try:
            calculated_fcff = (ebit * (1 - tax_rate)) + depreciation - cap_exp - change_in_nwc
            st.write(f"Calculated FCFF: {calculated_fcff:.2f}")
            
            # Verification
            if abs(fcff - calculated_fcff) < 0.01:
                st.success("The entered FCFF value is correct!")
            else:
                st.error("The entered FCFF value is incorrect.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Streamlit UI
def app():
    st.title('Stocks Page')
    page = st.sidebar.selectbox("Select Page", ["Stocks Description", "General Dividend Discount Model", 
                                                "Constant Dividend Growth Model", "Two Stage Dividend Growth Model", 
                                                "FCFF", "Free Cash Flows", "FCFF webscraping"])

    # Track the current page
    if "page" not in st.session_state:
        st.session_state.page = "Stocks Description"
    else:
        st.session_state.page = page

    if st.session_state.page == "Stocks Description":
        stocks_description()

    elif st.session_state.page == "General Dividend Discount Model":
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
    
    elif st.session_state.page == "FCFF webscraping":
        FCFF_webscraping()

if __name__ == "__main__":
    app()