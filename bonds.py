import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Define the bond value function
def bondvalue(facevalue, coupon_rate, maturity, ytm, payments_per_period):
    coupon_rate = coupon_rate / 100
    ytm = ytm / 100
    
    if ytm==0 or payments_per_period==0:
        return 0

    PVA = (facevalue * coupon_rate / payments_per_period * (1 - (1 + ytm / payments_per_period) ** (-payments_per_period * maturity))) / (ytm / payments_per_period)
    PA = facevalue * (1 + (ytm / payments_per_period)) ** (-payments_per_period * maturity)
    bondPrice = PVA + PA
    return bondPrice
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def bond_price(coupon_rate, initial_ytm, maturity, yield_change_range):
    periods = np.arange(1, maturity + 1)
    cash_flows = np.full(maturity, coupon_rate * 100)
    cash_flows[-1] += 100  # add principal repayment at maturity
    
    # Calculate bond price for given YTM
    bond_price = np.sum(cash_flows / (1 + initial_ytm / 100) ** periods)
    
    # Calculate bond prices for range of YTMs
    ytms = np.linspace(initial_ytm - yield_change_range, initial_ytm + yield_change_range, 100)
    prices = [np.sum(cash_flows / (1 + ytm / 100) ** periods) for ytm in ytms]
    
    return bond_price, ytms, prices


# Bond Description
def bonds_description():
    st.write("""
    ## Understanding Bonds
    A bond is a fixed income instrument that represents a loan made by an investor to a borrower (typically corporate or governmental). 
    A bond could be thought of as an I.O.U. between the lender and borrower that includes the details of the loan and its payments. 
    Bonds are used by companies, municipalities, states, and sovereign governments to finance projects and operations. 
    Owners of bonds are debtholders, or creditors, of the issuer.
    """)
    
# Bond Price vs YTM graph
def plot_bond_price_vs_ytm(facevalue, payments_per_period):
    # Explanation for users
    st.markdown("""
    ### Understanding the Relationship Between Bond Price and Yield to Maturity (YTM)
    
    - **Bond Price**: The price you pay to purchase the bond. This is influenced by the bond's coupon rate, maturity, and the market's required return (YTM).
    - **Yield to Maturity (YTM)**: The total return expected on a bond if held until maturity. It's expressed as an annual percentage rate.

    #### Key Concepts:                                                                                                                                                                                                                                   
    1. **Inverse Relationship**: Bond prices and YTM have an inverse relationship. When YTM increases, the present value of the bond's future cash flows decreases, leading to a lower bond price. Conversely, when YTM decreases, bond prices increase.
    
    2. **Coupon Rate Influence**: Higher coupon rates mean higher periodic interest payments, which increases the bond's price at a given YTM compared to bonds with lower coupon rates.
    
    3. **Maturity Impact**: Longer maturity bonds are more sensitive to changes in YTM. As maturity increases, the bond's price becomes more volatile with changes in YTM.

    4. **Discount and Premium Bonds**:
        - If the bond's coupon rate is higher than the YTM, the bond will trade at a premium (above face value).
        - If the bond's coupon rate is lower than the YTM, the bond will trade at a discount (below face value).
    
    #### Practical Application:
    - **Investment Strategy**: Investors use YTM to compare bonds with different coupon rates and maturities to determine which bonds offer the best return for the given level of risk.
    - **Market Conditions**: Understanding how bond prices react to changes in interest rates (which influence YTM) helps investors make informed decisions about buying or selling bonds in response to economic conditions.
    """)
    col1, col2 = st.columns(2)
    with col1:
        coupon_rate = st.slider('Coupon Rate (%)', 1.0, 20.0, 8.0, key="coupon_rate")
    with col2:
        maturity = st.slider('Maturity (years)', 1.0, 30.0, 10.0, key="maturity")

    ytms = list(range(1, 21))
    bond_prices = [bondvalue(facevalue, coupon_rate, maturity, ytm, payments_per_period) for ytm in ytms]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(ytms, bond_prices)
    ax.set_title('Variation of Bond Price and YTM')
    ax.set_xlabel('Yield to Maturity (%)')
    ax.set_ylabel('Bond Price')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 2000)
    ax.grid(True)
    
     # Set smaller font sizes for axis labels and ticks
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=5)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    st.pyplot(fig)

    

def generate_graph_ytm_bonds(facevalue, maturity, coupon_rate, payments_per_period):
    x = [i + 1 for i in range(10)]
    y = [bondvalue(facevalue, coupon_rate, maturity, ytm, payments_per_period) for ytm in x]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid(True)

    string = f'The parameters for the graph are:\nCoupon rate is ({coupon_rate}) and Maturity is ({maturity})\nand YTM is varying from 1 to 10'
    plt.text(0.95, 0.95, string, fontsize=10, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel('YTM')
    ax.set_ylabel('Price of Bond')
    ax.set_title('Price of Bond vs YTM')

    st.pyplot(fig)

# Percentage Change Graph
def plot_percentage_change(facevalue, payments_per_period, k):
    st.markdown("""
    Let's see how four different bonds with varying coupon rates and maturities respond to changes in YTM. Use the sliders to experiment with different parameters for each bond and observe the resulting percentage changes in their prices.
    """)

    st.write("Adjust the parameters for the four bonds:")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # Arrange sliders in columns
    col1, col2, col3, col4 = st.columns(4)

    # Define parameters for multiple bonds
    bonds = [
        {'name': 'Bond 1','coupon_rate': col1.slider('Coupon Rate (%) - Bond 1', min_value=0.1, max_value=10.0, step=0.1, value=5.0),
         'initial_ytm': col1.slider('Initial YTM (%) - Bond 1', min_value=0.1, max_value=10.0, step=0.1, value=2.0),
         'maturity': col1.slider('Maturity (years) - Bond 1', min_value=1, max_value=30, step=1, value=10),
         'yield_change_range': col1.slider('Yield Change Range (%) - Bond 1', min_value=0.1, max_value=5.0, step=0.1, value=2.0)},
    
        {'name': 'Bond 2','coupon_rate': col2.slider('Coupon Rate (%) - Bond 2', min_value=0.1, max_value=10.0, step=0.1, value=4.0),
         'initial_ytm': col2.slider('Initial YTM (%) - Bond 2', min_value=0.1, max_value=10.0, step=0.1, value=3.0),
         'maturity': col2.slider('Maturity (years) - Bond 2', min_value=1, max_value=30, step=1, value=15),
         'yield_change_range': col2.slider('Yield Change Range (%) - Bond 2', min_value=0.1, max_value=5.0, step=0.1, value=1.5)},
    
        {'name': 'Bond 3','coupon_rate': col3.slider('Coupon Rate (%) - Bond 3', min_value=0.1, max_value=10.0, step=0.1, value=6.0),
         'initial_ytm': col3.slider('Initial YTM (%) - Bond 3', min_value=0.1, max_value=10.0, step=0.1, value=4.0),
         'maturity': col3.slider('Maturity (years) - Bond 3', min_value=1, max_value=30, step=1, value=20),
         'yield_change_range': col3.slider('Yield Change Range (%) - Bond 3', min_value=0.1, max_value=5.0, step=0.1, value=3.0)},
    
        {'name': 'Bond 4','coupon_rate': col4.slider('Coupon Rate (%) - Bond 4', min_value=0.1, max_value=10.0, step=0.1, value=3.5),
         'initial_ytm': col4.slider('Initial YTM (%) - Bond 4', min_value=0.1, max_value=10.0, step=0.1, value=2.5),
         'maturity': col4.slider('Maturity (years) - Bond 4', min_value=1, max_value=30, step=1, value=5),
         'yield_change_range': col4.slider('Yield Change Range (%) - Bond 4', min_value=0.1, max_value=5.0, step=0.1, value=2.5)}
         ]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each bond on the same plot
    for bond in bonds:
        bond_price_val, ytms, prices = bond_price(bond['coupon_rate'], bond['initial_ytm'], bond['maturity'], bond['yield_change_range'])
        ax.plot(ytms, prices, label=f'{bond["name"]} (Coupon Rate: {bond["coupon_rate"]}%, Maturity: {bond["maturity"]} years)')

    # Customize plot
    ax.set_xlabel('Yield to Maturity (%)')
    ax.set_ylabel('Bond Price')
    ax.legend()
    st.pyplot(fig)

def three_graph_merge(facevalue, payments_per_period):
    x = [i for i in range(10)]
    y1 = []
    y2 = []
    y3 = []
    for i in x:
        y1.append(bondvalue(facevalue, 10, i, 8, payments_per_period))
        y2.append(bondvalue(facevalue, 6, i, 8, payments_per_period))
        y3.append(bondvalue(facevalue, 8, i, 8, payments_per_period))

    z = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot the initial line graphs
    ax.plot(z, y1, label="Premium Bond")
    ax.plot(z, y2, label="Discount Bond", linestyle='dashed')
    ax.plot(z, y3, label="Par Bond", linestyle='dotted')
    # Set labels and title
    ax.set_xlabel('Years to Maturity')
    ax.set_ylabel('Bond Price')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

def three_graph_merge_2(facevalue, payments_per_period, cr1, ytm1, cr2, ytm2):
    x = [i for i in range(15)]
    y1 = []
    y2 = []
    y3 = []
    for i in x:
        y3.append(bondvalue(facevalue, 8, i, 8, payments_per_period))
        y1.append(bondvalue(facevalue, cr1, i, ytm1, payments_per_period))
        y2.append(bondvalue(facevalue, cr2, i, ytm2, payments_per_period))

    z = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    fig, ax = plt.subplots(figsize=(15, 10))
    # Plot the initial line graphs
    ax.plot(z, y1, label=f"Bond 1: Coupon Rate {cr1}%, YTM {ytm1}%")
    ax.plot(z, y2, label=f"Bond 2: Coupon Rate {cr2}%, YTM {ytm2}%")
    ax.plot(z, y3, label=f"Bond 3: Coupon Rate 8%, YTM 8%")
    # Set labels and title
    ax.set_xlabel('Years to Maturity')
    ax.set_ylabel('Bond Price')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.markdown("""
    ### Bond Price vs Maturity, Coupon Rate, and Yield to Maturity (YTM)
    
    - **Maturity**: Longer maturity generally leads to higher bond prices due to higher present value of future cash flows.
    - **Coupon Rate**: Higher coupon rates usually result in higher bond prices, as they provide higher periodic interest payments.
    - **Yield to Maturity (YTM)**: Lower YTM tends to increase bond prices, reflecting higher present value of future cash flows relative to the lower discount rate.
   
    #### Types of Bonds:
    1. **Par Bonds**: Bonds that are trading at their face value. This typically happens when the coupon rate equals the YTM.
    2. **Discount Bonds**: Bonds that are trading below their face value. This occurs when the coupon rate is lower than the YTM, indicating that the bond is offering less than the market's required return.
    3. **Premium Bonds**: Bonds that are trading above their face value. This happens when the coupon rate is higher than the YTM, indicating that the bond is offering more than the market's required return.
    
    """)

CUSTOM_CSS = """
<style>
    /* Adjust slider label font size */
    div[class*="stSlider"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 10px !important; /* Adjust as needed */
    }

    /* Adjust slider value font size */
    div[class*="stSlider"] > div > div[class*="slider-value"] {
        font-size: 1px !important; /* Adjust as needed */
</style>
"""

def app():
    st.title("Bond Analysis")

    # Sidebar for page navigation
    page = st.sidebar.selectbox("Select Page", ["Bonds Description", "Generate graph Ytm vs Bond price", "Bond Price vs YTM", "Percentage Change Graph", "Three Graph Merge-1", "Three Graph Merge-2"])

    # Track the current page
    if "page" not in st.session_state:
        st.session_state.page = "Bonds Description"
    else:
        st.session_state.page = page

    # Display the appropriate page based on the selected page
    facevalue = 1000
    payments_per_period = 1

    if st.session_state.page == "Bonds Description":
        bonds_description()
    elif st.session_state.page == "Generate graph Ytm vs Bond price":
        st.write("Enter the term of the bond and coupon rate in the format 'maturity,couponrate'.")
        user_input = st.text_input("Input", "10,8")
        if st.button("Generate Graph"):
            try:
                maturity, coupon_rate = map(int, user_input.split(','))
                generate_graph_ytm_bonds(facevalue, maturity, coupon_rate, payments_per_period)
            except ValueError:
                st.error("Invalid input format. Please enter values in the format 'maturity,couponrate'.")

    elif st.session_state.page == "Bond Price vs YTM":
        plot_bond_price_vs_ytm(facevalue, payments_per_period)
    elif st.session_state.page == "Percentage Change Graph":
        plot_percentage_change(1000, 1, 5)
    elif st.session_state.page == "Three Graph Merge-1":
        three_graph_merge(facevalue, payments_per_period)
    elif st.session_state.page == "Three Graph Merge-2":
        col1, col2 = st.columns(2)
        with col1:
            cr1 = st.sidebar.slider('Coupon Rate (%) of Bond 1', 1.0, 15.0, 11.0)
            ytm1 = st.sidebar.slider('YTM (%) of Bond 1', 1.0, 15.0, 5.0)
        with col2:
            cr2 = st.sidebar.slider('Coupon Rate (%) of Bond 2', 1.0, 15.0, 3.0)
            ytm2 = st.sidebar.slider('YTM (%) of Bond 2', 1.0, 15.0, 10.0)
        three_graph_merge_2(facevalue, payments_per_period, cr1, ytm1, cr2, ytm2)
    
        
if __name__ == "__main__":
    app()
