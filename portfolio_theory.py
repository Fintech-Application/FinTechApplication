import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from fredapi import Fred
import pandas_datareader.data as web
from pandas_datareader.famafrench import FamaFrenchReader
from datetime import datetime
import numpy as np
import yfinance as yf

DEFAULT_API_KEY = "c88d55f2072ae1cb0915f9c827a354ce"

def portfolio_description():
    st.write("""
    Portfolio theory, also known as Modern Portfolio Theory (MPT), is a framework developed by Harry Markowitz in the 1950s that aims to maximize expected return for a given level of risk, or minimize risk for a given level of expected return. Key principles include:
    - **Diversification**: Spreading investments across different asset classes to reduce overall risk.
    - **Risk and Return**: Balancing the trade-off between risk (volatility) and return (profitability) by selecting assets with different risk profiles.
    - **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a given level of risk, or the lowest risk for a given level of return.
    - **Asset Allocation**: Determining the mix of assets (stocks, bonds, etc.) in a portfolio based on an investor's risk tolerance and investment goals.
    - **Capital Market Line (CML)**: Represents the combination of a risk-free asset and the efficient frontier, offering optimal risk-return combinations.
    
    Portfolio theory provides a quantitative approach to constructing diversified investment portfolios that aim to achieve optimal risk-adjusted returns.
    """)

def inflation_vs_tbills():
    st.title('Inflation Rate vs T-Bills Rate Visualization')
    st.write("""
    This app allows you to visualize the Inflation Rate and T-Bills Rate over time.
    Please upload two Excel files with the following formats:
    1. **Inflation Data File**: Columns: `Year` (int) and `Value` (float)
    2. **T-Bills Data File**: Columns: `Year` (int) and `Value` (float)
    
    Make sure the columns are named exactly as `Year` and `Value`.
    """)
    # Provide download buttons for sample files
    col1, col2 = st.columns(2)
    with col1:
        with open("datasets/inflation.xlsx", "rb") as file:
            st.download_button(label="Download Inflation Data File",
                               data=file,
                               file_name='sample_inflation_data.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with col2:
        with open("datasets/t-bills.xlsx", "rb") as file:
            st.download_button(label="Download T-Bills Data File",
                               data=file,
                               file_name='sample_tbills_data.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    inflation_file = st.file_uploader("Upload the Inflation Data File (Excel)", type=["xlsx"])
    tbills_file = st.file_uploader("Upload the T-Bills Data File (Excel)", type=["xlsx"])

    if inflation_file is not None and tbills_file is not None:
        inflation_df = pd.read_excel(inflation_file)
        tbills_df = pd.read_excel(tbills_file)

        if 'Year' in inflation_df.columns and 'Value' in inflation_df.columns and 'Year' in tbills_df.columns and 'Value' in tbills_df.columns:
            inflation_df.set_index('Year', inplace=True)
            tbills_df.set_index('Year', inplace=True)

             # Input fields for year selection
            start_year = st.sidebar.number_input('Start Year', min_value=inflation_df.index.min(), max_value=inflation_df.index.max(), value=inflation_df.index.min())
            end_year = st.sidebar.number_input('End Year', min_value=inflation_df.index.min(), max_value=inflation_df.index.max(), value=inflation_df.index.max())

            # Filter data based on selected years
            inflation_df_filtered = inflation_df.loc[start_year:end_year]
            tbills_df_filtered = tbills_df.loc[start_year:end_year]

            plt.figure(figsize=(10, 6))
            plt.plot(inflation_df_filtered.index, inflation_df_filtered['Value'], label='Inflation Rate', linestyle='-')
            plt.plot(tbills_df_filtered.index, tbills_df_filtered['Value'], label='T-Bills Rate', linestyle='-')
            plt.title('Inflation Rate vs T-Bills Rate')
            plt.xlabel('Year')
            plt.ylabel('Value')
            plt.xticks(range(start_year, end_year + 1, 5))
            plt.legend()
            st.pyplot(plt)
        else:
            st.error("The uploaded files do not contain the required columns: 'Year' and 'Value'. Please check your files and try again.")
    else:
        st.warning("Please upload both the Inflation Data File and the T-Bills Data File to see the plot.")

def inflation_vs_tbills_api():
    fred = Fred(api_key=DEFAULT_API_KEY)

    # Streamlit app title
    st.title('T-bills and Inflation Line Graph Using FRED API')

    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    tbill_series_id = st.sidebar.text_input('T-bill FRED Series ID', 'RIFSGFSM03NA') #another ID to test is RIFSGFSM03NA 
    inflation_series_id = st.sidebar.text_input('Inflation FRED Series ID', 'CPIAUCSL')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('1954-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

    # Fetching data from FRED
    if tbill_series_id and inflation_series_id and start_date and end_date:
        tbill_data = fred.get_series(tbill_series_id, start_date, end_date)
        inflation_data = fred.get_series(inflation_series_id, start_date, end_date)

        # Convert to DataFrame
        df_tbill = pd.DataFrame(tbill_data, columns=['Value'])
        df_tbill.index = pd.to_datetime(df_tbill.index)
        df_tbill.reset_index(inplace=True)
        df_tbill.rename(columns={'index': 'Date'}, inplace=True)

        # Resample CPI data to annual frequency and compute the year-over-year percent change
        inflation_data = inflation_data.resample('A').mean()
        inflation_annual_pct_change = inflation_data.pct_change() * 100
        df_inflation = inflation_annual_pct_change.to_frame(name='Value')
        df_inflation.index = pd.to_datetime(df_inflation.index)
        df_inflation.reset_index(inplace=True)
        df_inflation.rename(columns={'index':'Date'},inplace=True)

        # Plotting the line graph
        plt.figure(figsize=(12, 8))
        plt.plot(df_tbill['Date'], df_tbill['Value'], linestyle='-', label='T-Bill')
        plt.plot(df_inflation['Date'], df_inflation['Value'], linestyle='-', label='Inflation')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'T-bills ({tbill_series_id}) and Inflation Data ({inflation_series_id}) from FRED API')
        plt.legend()
        plt.grid(True)

        # Display the plot
        st.pyplot(plt.gcf())

    # Instructions on using the app
    st.markdown("""
    ## Instructions:
    1. Enter the FRED Series ID for the T-bills data (default is 'DTB3' for 3-Month Treasury Bill).
    2. Enter the FRED Series ID for the Inflation data (default is 'CPIAUCSL' for Consumer Price Index for All Urban Consumers).
    3. Select the start and end dates for the data range.
    4. The line graph will be generated based on the input parameters.
    """)

def small_vs_large_portfolio():
    st.title('Small vs Large Stock Portfolio (1926-2023)')
    st.write("""
    ### Instructions:
    1. Upload a CSV file with the following columns: **Year, Large Stock Total Return, Small Stock Total Return**.
    2. Ensure that the **Small Stock Total Return** and **Large Stock Total Return** columns are formatted as decimal values.
    3. The data should cover the period from 1926 to 2023.
    """)

    col1, col2 = st.columns(2)
    with col1:
        with open("datasets/SBBI_Annual-data.csv", "rb") as file:
            st.download_button(label="Download Stocks Data File",
                               data=file,
                               file_name='SBBI_annual-data.csv',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        stocks = pd.read_csv(uploaded_file)
        
        if all(col in stocks.columns for col in ['Year', 'Large Stock Total Return', 'Small Stock Total Return']):
            # Create the DataFrame
            df = stocks[['Year', 'Large Stock Total Return', 'Small Stock Total Return']]
            df.rename(columns={'Large Stock Total Return': 'large stocks', 'Small Stock Total Return': 'small stocks'}, inplace=True)
            df.set_index('Year', inplace=True)

            # Input fields for year selection
            start_year = st.sidebar.number_input('Start Year', min_value=int(df.index.min()), max_value=int(df.index.max()), value=int(df.index.min()))
            end_year = st.sidebar.number_input('End Year', min_value=int(df.index.min()), max_value=int(df.index.max()), value=int(df.index.max()))

            # Filter data based on selected year range
            filtered_df = df[(df.index >= start_year) & (df.index <= end_year)]

            years = filtered_df.index
            small_stock_portfolio = filtered_df['small stocks']
            large_stock_portfolio = filtered_df['large stocks']

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(years, small_stock_portfolio, linestyle='-', color='b', label='Small Stock (Lo 30)')
            ax.plot(years, large_stock_portfolio, linestyle='-', color='r', label='Large Stock (Hi 30)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Annual Stock Returns')
            ax.set_title('Small vs Large Stock Portfolio (1926-2023)')
            ax.legend()
            ax.grid(True)
            ax.set_xticks(range(start_year, end_year + 1, 6))
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("The uploaded CSV file does not have the required columns.")
    else:
        st.warning("Please upload a CSV file to display the plot.")

    st.markdown("""
    ## Explanation:
    
    1. **Data Columns:**
       - **Year:** The year of the return data.
       - **Large Stock Total Return:** Total annual return for large stocks (in decimal form, e.g., 0.05 for 5%).
       - **Small Stock Total Return:** Total annual return for small stocks (in decimal form, e.g., 0.05 for 5%).
    
    2. **Annual Returns Calculation:**
       - The annual return for each stock portfolio is used directly from the data. The data is visualized over the selected period.
    
    3. **Plot Interpretation:**
       - The plot shows the annual returns of small vs. large stock portfolios over time.
       - **Small Stock Portfolio (Lo 30):** Represented by the blue line.
       - **Large Stock Portfolio (Hi 30):** Represented by the red line.
    """)
def small_vs_large_portfolio_api():
    st.title('Small vs Large Stock Portfolio (1926-2023)')

    st.write("""
    ### Instructions:
    1. Select the date range for which you want to display the stock portfolio returns.
    2. The data is retrieved directly from the Fama-French Data Library.
    """)

    # Date range selection
    start_date = st.date_input("Start Date", datetime(1926, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 12, 31))

    if start_date > end_date:
        st.error('Error: End date must fall after start date.')
        return

    # Fetch Fama-French data using FamaFrenchReader
    def get_fama_french_data():
        ff_reader = FamaFrenchReader("Portfolios_Formed_on_ME", start=start_date, end=end_date)
        df = ff_reader.read()
        return df[0]  # First element of the tuple is the DataFrame we want

    # Fetch and display the data
    if st.button("Fetch Data"):
        try:
            df = get_fama_french_data()
            
            # Convert index to Timestamps if it's in Period format
            if isinstance(df.index, pd.PeriodIndex):
                df.index = df.index.to_timestamp()

            # Resample the data to annual frequency using the compound return formula
            df_annual = df.resample('Y').apply(lambda x: (1 + x / 100).prod() - 1)
            
            small_stock_portfolio = df_annual['Lo 30']
            large_stock_portfolio = df_annual['Hi 30']
            
            # Plotting the data
            fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size
            ax.plot(small_stock_portfolio.index, small_stock_portfolio, linestyle='-', color='b', label='Small Stock (Lo 30)')
            ax.plot(large_stock_portfolio.index, large_stock_portfolio, linestyle='-', color='r', label='Large Stock (Hi 30)')
            ax.set_xlabel('Year')
            ax.set_ylabel('Annual Stock Returns')
            ax.set_title('Small vs Large Stock Portfolio (1926-2023)')
            ax.legend()
            ax.grid(True)

            # Set x-axis ticks every 5 years within the selected range
            year_range = range(start_date.year, end_date.year + 1, 5)
            ax.set_xticks(pd.date_range(start=start_date, end=end_date, freq='5Y'))
            ax.set_xticklabels(year_range)

            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Select a date range and click 'Fetch Data' to display the plot.")
    
    st.markdown("""
    ## Conceptual Understanding:
    
    1. **Data Source:**
       - The data for small and large stock portfolios is retrieved from the Fama-French Data Library.
    
    2. **Small vs. Large Stock Portfolios:**
       - **Small Stock Portfolio (Lo 30):** Represents the 30% of the market capitalization of the smallest stocks.
       - **Large Stock Portfolio (Hi 30):** Represents the 30% of the market capitalization of the largest stocks.
    
    3. **Annual Returns Calculation:**
       - The annual returns are calculated using the compound return formula. In LaTeX, this formula is:
    """)
    st.latex(r'''
        R_{\text{annual}} = \left(\prod_{i=1}^{n} (1 + R_i)\right) - 1
    ''')
    st.markdown("""
    Where \( Ri \) represents the monthly returns.

    4. **Plot Interpretation:**
       - The plot shows the annual returns of the small vs. large stock portfolios over the selected period.
       - The small stock portfolio (blue line) and the large stock portfolio (red line) are compared to observe performance trends over time.
    """)

def Annual_tbill_returns():
    st.title('Frequency Distribution of Annual T-bill Returns (1954-2023)')
    col1, col2 = st.columns(2)
    with col1:
        with open("datasets/t-bills.xlsx", "rb") as file:
            st.download_button(label="Download T-bills Data File",
                               data=file,
                               file_name='t-bills.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your T-bills Excel file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        if 'Year' in df.columns and 'Value' in df.columns:
            df.set_index('Year', inplace=True)
            # Input fields for year selection
            start_year = st.sidebar.number_input('Start Year', min_value=df.index.min(), max_value=df.index.max(), value=df.index.min())
            end_year = st.sidebar.number_input('End Year', min_value=df.index.min(), max_value=df.index.max(), value=df.index.max())
        
            df_filtered = df.loc[start_year:end_year]

            # Defining bins and calculating frequency distribution
            bins = list(range(-45, 65, 5))
            df_filtered['Binned'] = pd.cut(df['Value'], bins=bins, right=False)
            freq = df_filtered['Binned'].value_counts().sort_index()

            # Converting counts to percentages
            freq_percentage = (freq / freq.sum()) * 100

            # Plotting the frequency distribution
            plt.figure(figsize=(12, 8))
            plt.bar(freq_percentage.index.astype(str), freq_percentage, width=0.8, edgecolor='black')
            plt.xlabel('Return Ranges (%)')
            plt.ylabel('Percentage of Observations (%)')
            plt.title('Frequency Distribution of Annual T-bill Returns (1954-2023)')
            plt.xticks(rotation=45)
            plt.grid(axis='y')

            # Display the plot
            st.pyplot(plt.gcf())
        else:
            st.error("The uploaded files do not contain the required columns: 'Year' and 'Value'. Please check your files and try again.")
    else:
        st.warning("Please upload the T-Bills Data File to see the plot.")


    # Instructions on file format
    st.markdown("""
    ## Instructions:
    1. **Download Template:** You can download the data file from the link provided .
    2. **Upload File:** The uploaded file should be in Excel format (`.xlsx`) and contain a single sheet.
    3. **File Columns:** The sheet should contain columns labeled `Year` and `Value`.
       - The `Year` column should have the years of the data.
       - The `Value` column should have the annual T-bill returns.
    4. **Select Years:** Use the sidebar to select the start and end years for the data range you want to analyze.
    5. **View Plot:** The bar graph will show the frequency distribution of annual T-bill returns for the selected years.

    ## Conceptual Understanding:
    1. **Frequency Distribution:**
       - The annual T-bill returns are divided into specified bins ranging from -45% to 60% with a step of 5%.
       - The frequency of observations within each bin is counted and converted to a percentage.
    
    2. **Plot Interpretation:**
       - The bar graph displays the percentage of observations within each return range over the specified period.
    """)

    st.write("""
    ## Formulas:
    """)

    st.latex(r"""
    \text{Frequency Distribution} = \left( \frac{\text{Count in Bin}}{\text{Total Count}} \right) \times 100
    """)


def Annual_tbill_returns_API():
    fred = Fred(api_key=DEFAULT_API_KEY)

    # Streamlit app title
    st.title('T-bills Bar Graph Using FRED API')

    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    series_id = st.sidebar.text_input('FRED Series ID', 'DTB3')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('1954-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

    # Fetching data from FRED
    if series_id and start_date and end_date:
        data = fred.get_series(series_id, start_date, end_date)

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['Value'])
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        # Calculate frequency distribution
        bins = np.arange(-45, 65, 5)  # Define bins for the histogram
        labels = [f'{i}-{i+5}%' for i in bins[:-1]]  # Labels for each bin
        binned_data = pd.cut(df['Value'], bins=bins, labels=labels, right=True)
        freq_distribution = binned_data.value_counts(normalize=True) * 100  # Calculate frequency distribution in percentage

        # Plotting the bar graph
        plt.figure(figsize=(12, 8))
        freq_distribution.sort_index().plot(kind='bar', edgecolor='black')
        plt.xlabel('Return Ranges (%)')
        plt.ylabel('Percentage of Observations (%)')
        plt.title(f'T-bills Data ({series_id}) from FRED API')
        plt.grid(True)

        # Display the plot
        st.pyplot(plt.gcf())

    # Instructions on using the app
    st.markdown("""
    ## Instructions and Conceptual Understanding:

    1. **Input Parameters:**
       - Use the sidebar to enter the FRED Series ID (e.g., 'DTB3' for 3-Month Treasury Bill) and select the start and end dates to fetch data.

    2. **Data Fetching:**
       - The app retrieves historical T-bill data for the specified series from the FRED API.
       - The fetched data includes the interest rates for the specified period.

    3. **Frequency Distribution Calculation:**
       - The T-bill rates are divided into specified bins ranging from -45% to 60% with a step of 5%.
       - The frequency of observations within each bin is counted and converted to a percentage.

    4. **Plot Interpretation:**
       - The bar graph displays the percentage of observations within each return range over the specified period.
    """)
    
    st.write("""
    ## Formulas:
    """)

    st.latex(r"""
    \text{Frequency Distribution} = \left( \frac{\text{Count in Bin}}{\text{Total Count}} \right) \times 100
    """)

def Annual_SP500_returns_API():
    fred = Fred(api_key=DEFAULT_API_KEY)

    # Streamlit app title
    st.title('Frequency Distribution of Annual S&P 500 Returns (1927-2023)')

    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    # series_id = st.sidebar.text_input('FRED Series ID', 'SP500')
    ticker = st.sidebar.text_input('Ticker Symbol', '^GSPC')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('1927-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

    # # Fetching S&P 500 data from FRED
    # sp500_data = fred.get_series(series_id, observation_start=start_date.strftime('%Y-%m-%d'), observation_end=end_date.strftime('%Y-%m-%d'))

    # Step 1: Download the market index data (S&P 500 in this example)
    sp500_data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate annual returns
    sp500_data = sp500_data['Adj Close'].resample('Y').last()  # Resample to annual data
    annual_returns = sp500_data.pct_change().dropna() * 100  # Calculate annual returns in percentage

    # Create bins and labels
    bins = np.arange(-45, 65, 5)  # Create bins from -45 to 60 with a step of 5

    # Cut the annual returns into the specified bins
    binned_returns = pd.cut(annual_returns, bins=bins, right=False)

    # Count the number of observations in each bin
    freq = binned_returns.value_counts().sort_index()

    # Convert counts to percentages
    freq_percentage = (freq / freq.sum()) * 100

    # Plotting the bar graph
    plt.figure(figsize=(12, 8))
    plt.bar(freq_percentage.index.astype(str), freq_percentage, width=0.8, edgecolor='black')
    plt.xlabel('Return Ranges (%)')
    plt.ylabel('Percentage of Observations (%)')
    plt.title('Frequency Distribution of Annual S&P 500 Returns (1927-2023)')
    plt.xticks(rotation=45)
    plt.yticks(np.arange(0, freq_percentage.max() + 5, 5))
    plt.grid(axis='y')

    # Display the plot
    st.pyplot(plt.gcf())

    st.write("""
    ## Instructions and Conceptual Understanding:

    1. **Input Parameters:**
       - Use the sidebar to enter the ticker symbol (e.g., '^GSPC') and select the start and end dates to fetch data.

    2. **Data Fetching:**
       - The app retrieves historical closing prices of the specified market index (S&P 500) from Yahoo Finance.

    3. **Annual Returns Calculation:**
       - The annual returns are calculated as the percentage change in adjusted closing prices, resampled to the last trading day of each year.

    4. **Binning and Frequency Calculation:**
       - The annual returns are divided into specified bins ranging from -45% to 60% with a step of 5%.
       - The frequency of observations within each bin is counted and converted to a percentage.

    5. **Plot Interpretation:**
       - The bar graph displays the percentage of observations within each return range over the specified period.
    """)

    st.write("""
    ## Formulas:
    """)

    st.latex(r"""
    \text{Annual Return} = \left( \frac{\text{Adj Close}_{\text{end}}}{\text{Adj Close}_{\text{start}}} \right) - 1 \times 100
    """)

def Annualized_Std_Dev_Sp500():
    fred = Fred(api_key=DEFAULT_API_KEY)

    # Streamlit app title
    st.title('Annualized Standard Deviation of Monthly Excess Returns of S&P 500')

    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    ticker = st.sidebar.text_input('Ticker Symbol', '^GSPC')
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('1927-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))

    # Step 1: Download the market index data (S&P 500 in this example)
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Step 2: Calculate daily returns
    data['Daily_Return'] = data['Adj Close'].pct_change()

    # Step 3: Resample to monthly frequency
    monthly_data = data.resample('M').last()

    # Step 4: Calculate monthly returns
    monthly_data['Monthly_Return'] = monthly_data['Adj Close'].pct_change()

    # Step 5: Fetch historical risk-free rate data from FRED
    risk_free_rate_data = fred.get_series('TB3MS', observation_start=start_date.strftime('%Y-%m-%d'), observation_end=end_date.strftime('%Y-%m-%d'))
    risk_free_rate_data = risk_free_rate_data.resample('M').last() / 100  # Convert from percentage to decimal

    # Align the risk-free rate data with the monthly data index
    risk_free_rate_data = risk_free_rate_data.reindex(monthly_data.index, method='ffill')

    # Step 6: Calculate excess returns
    monthly_data['Excess_Return'] = monthly_data['Monthly_Return'] - risk_free_rate_data

    # Step 7: Calculate rolling standard deviation of excess returns
    rolling_window = 12  # 1 year rolling window
    monthly_data['Rolling_Std_Dev'] = monthly_data['Excess_Return'].rolling(window=rolling_window).std()

    # Step 8: Annualize standard deviation
    monthly_data['Annualized_Std_Dev'] = monthly_data['Rolling_Std_Dev'] * np.sqrt(12) * 100

    # Step 9: Plot the annualized standard deviation
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index, monthly_data['Annualized_Std_Dev'], label='Annualized Std Dev of Excess Returns', color='r')
    plt.xlabel('Year')
    plt.ylabel('Annualized Std Deviation (%)')
    plt.title('Annualized Standard Deviation of Monthly Excess Returns of S&P 500')
    plt.legend()
    plt.grid(True)

    # Customize x-axis ticks
    plt.xticks(pd.date_range(start=start_date, end=end_date, freq='5Y'), rotation=45)  # Adjusting x-axis ticks every 5 years
    plt.tight_layout()

    # Display the plot
    st.pyplot(plt.gcf())

    st.write("""
    ## Instructions and Conceptual Understanding:
    
    1. **Input Parameters:**
       - Use the sidebar to enter the ticker symbol (e.g., '^GSPC') and select the start and end dates to fetch data.
    
    2. **Data Fetching:**
       - The app retrieves historical daily closing prices of the specified market index (S&P 500) from Yahoo Finance.
       - Daily returns are calculated as the percentage change in adjusted closing prices.
    
    3. **Resampling to Monthly Frequency:**
       - Daily data is resampled to monthly frequency, taking the last trading day's closing price of each month.
       - Monthly returns are computed as the percentage change in adjusted closing prices month-over-month.
    
    4. **Historical Risk-Free Rate:**
       - Historical risk-free rate data (3-month Treasury Bill rate) is obtained from the Federal Reserve Economic Data (FRED) API.
       - The rate is resampled to monthly frequency and converted from percentage to decimal form.
    
    5. **Excess Returns Calculation:**
       - Excess returns are calculated as the difference between monthly market returns and the historical risk-free rate.
    
    6. **Rolling Standard Deviation:**
       - A rolling 1-year (12 months) standard deviation of excess returns is computed to measure volatility.
    
    7. **Annualized Standard Deviation:**
       - The rolling standard deviation is annualized by multiplying it by the square root of 12 (for monthly data) and 100 to convert to percentage.
    
    8. **Plot Interpretation:**
       - The line graph displays the annualized standard deviation of monthly excess returns over time.
       - The subtitle indicates the average risk-free rate used in the calculations.
    """)

    st.write("""
    ## Formulas:
    """)

    st.latex(r"""
    \text{Daily Return} = \frac{\text{Adj Close}_t}{\text{Adj Close}_{t-1}} - 1
    """)

    st.latex(r"""
    \text{Monthly Return} = \frac{\text{Adj Close}_t}{\text{Adj Close}_{t-1}} - 1
    """)

    st.latex(r"""
    \text{Excess Return} = \text{Monthly Return} - \text{Risk-Free Rate}
    """)

    st.latex(r"""
    \text{Rolling Std Dev} = \sqrt{\frac{\sum_{i=1}^N (R_i - \bar{R})^2}{N-1}}
    """)

    st.latex(r"""
    \text{Annualized Std Dev} = \text{Rolling Std Dev} \times \sqrt{12} \times 100
    """)
def growth_of_dollar():
    st.title('Growth of $1 Investment Over Time (1926 - Present)')

    col1, col2 = st.columns(2)
    with col1:
        with open("datasets/SBBI_Annual-data.csv", "rb") as file:
            st.download_button(label="Download SBBI Data File",
                               data=file,
                               file_name='SBBI_data-Annual.csv',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if 'Year' in data.columns:
            # Calculate cumulative returns for each investment category
            data['Cumulative Small Stock'] = (1 + data['Small Stock Total Return']).cumprod()
            data['Cumulative SP500'] = (1 + data['Large Stock Total Return']).cumprod()
            data['Cumulative IT Bonds'] = (1 + data['IT 5Y GovtBonds TR']).cumprod()
            data['Cumulative 30 Day TBill'] = (1 + data['30 Day TBill TR']).cumprod()
            data['Cumulative Inflation'] = (1 + data['Inflation']).cumprod()

            # Input fields for year selection
            min_year = st.sidebar.number_input('Start Year', min_value=int(data['Year'].min()), max_value=int(data['Year'].max()), value=int(data['Year'].min()))
            max_year = st.sidebar.number_input('End Year', min_value=int(data['Year'].min()), max_value=int(data['Year'].max()), value=int(data['Year'].max()))

            # Filter data based on selected year range
            data_filtered = data[(data['Year'] >= min_year) & (data['Year'] <= max_year)]

            # Normalize data to start at $1 for the selected year range
            start_index = data_filtered.index[0]
            data_filtered['Cumulative Small Stock'] /= data_filtered['Cumulative Small Stock'].iloc[0]
            data_filtered['Cumulative SP500'] /= data_filtered['Cumulative SP500'].iloc[0]
            data_filtered['Cumulative IT Bonds'] /= data_filtered['Cumulative IT Bonds'].iloc[0]
            data_filtered['Cumulative 30 Day TBill'] /= data_filtered['Cumulative 30 Day TBill'].iloc[0]
            data_filtered['Cumulative Inflation'] /= data_filtered['Cumulative Inflation'].iloc[0]

            # Prepare the plot
            plt.figure(figsize=(15, 8))

            # Plot each investment's cumulative growth on a log scale
            plt.plot(data_filtered['Year'], data_filtered['Cumulative Small Stock'], label='Small Company Stocks')
            plt.plot(data_filtered['Year'], data_filtered['Cumulative SP500'], label='S&P 500')
            plt.plot(data_filtered['Year'], data_filtered['Cumulative IT Bonds'], label='Intermediate Bonds')
            plt.plot(data_filtered['Year'], data_filtered['Cumulative 30 Day TBill'], label='30 Day T-Bills')
            plt.plot(data_filtered['Year'], data_filtered['Cumulative Inflation'], label='Inflation')

            # Annotate the final value on the plot
            final_year = data_filtered['Year'].iloc[-1]
            plt.text(final_year, data_filtered['Cumulative Small Stock'].iloc[-1], f"${data_filtered['Cumulative Small Stock'].iloc[-1]:.2f}", fontsize=10)
            plt.text(final_year, data_filtered['Cumulative SP500'].iloc[-1], f"${data_filtered['Cumulative SP500'].iloc[-1]:.2f}", fontsize=10)
            plt.text(final_year, data_filtered['Cumulative IT Bonds'].iloc[-1], f"${data_filtered['Cumulative IT Bonds'].iloc[-1]:.2f}", fontsize=10)
            plt.text(final_year, data_filtered['Cumulative 30 Day TBill'].iloc[-1], f"${data_filtered['Cumulative 30 Day TBill'].iloc[-1]:.2f}", fontsize=10)
            plt.text(final_year, data_filtered['Cumulative Inflation'].iloc[-1], f"${data_filtered['Cumulative Inflation'].iloc[-1]:.2f}", fontsize=10)

            # Set the scale and labels
            plt.yscale('log')
            plt.xlabel('Year')
            plt.ylabel('Growth of $1 Investment')
            plt.title('Growth of $1 Investment Over Time')
            plt.legend()

            # Customize the y-axis to show specific dollar amounts
            y_ticks = [1, 10, 100, 1000, 10000, 100000]
            plt.yticks(y_ticks, ['$1', '$10', '$100', '$1k', '$10k', '$100k'])
            plt.xticks(range(min_year, max_year + 1, 6))

            # Show the plot
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            st.pyplot(plt)

            
        else:
            st.write("The uploaded file does not contain the required 'Year' column.")
    else:
        st.warning("Please upload a CSV file.")
# Add description points for the users
    st.write("""
             ### Instructions:
             1. **Download SBBI Data File**: Click the button to download a sample data file if you don't have your own.
             2. **Upload CSV File**: Use the file uploader to upload your CSV file containing the required data.
             3. **Select Year Range**: Use the input fields in the sidebar to select the start and end years for your analysis.
             4. **View Plot**: Once the data is uploaded and the year range is selected, a plot will be generated showing the growth of $1 investment over time across different investment categories.
             """)

    st.write("""
### The Growth of a $1 Investment Over Time

#### 1. Introduction to Compound Interest

Compound interest is the fundamental principle that drives the growth of investments over time. It refers to the process by which an investment grows not only on the initial principal but also on the accumulated interest from previous periods. The formula for compound interest is expressed as:
""")

    st.latex(r'''
A = P \left(1 + \frac{r}{n}\right)^{nt}
''')

    st.write("""
where:
- \( A \) represents the amount of money accumulated after \( n \) years, including interest.
- \( P \) is the principal amount (the initial amount of money).
- \( r \) is the annual interest rate (as a decimal).
- \( n \) is the number of times interest is compounded per year.
- \( t \) is the time the money is invested for, in years.

Compound interest allows investments to grow exponentially over time, as interest is earned on both the initial principal and the previously accumulated interest.
""")
    st.write("""
### Example Calculation of Compound Interest
Let's calculate the growth of a $1 investment over three years with an annual interest rate of 5%, compounded once per year.

#### Year 1
""")
    st.latex(r'''
t = 1
''')
    st.latex(r'''
A_1 = 1 \left(1 + \frac{0.05}{1}\right)^{1 \cdot 1}
''')
    st.latex(r'''
A_1 = 1 \left(1 + 0.05\right)^1
''')
    st.latex(r'''
A_1 = 1 \cdot 1.05
''')
    st.latex(r'''
A_1 = 1.05
''')

    st.write("""
#### Year 2
""")
    st.latex(r'''
t = 2
''')
    st.latex(r'''
A_2 = 1 \left(1 + \frac{0.05}{1}\right)^{1 \cdot 2}
''')
    st.latex(r'''
A_2 = 1 \left(1 + 0.05\right)^2
''')
    st.latex(r'''
A_2 = 1 \cdot 1.1025
''')
    st.latex(r'''
A_2 = 1.1025
''')

    st.write("""
#### Year 3
""")
    st.latex(r'''
t = 3
''')
    st.latex(r'''
A_3 = 1 \left(1 + \frac{0.05}{1}\right)^{1 \cdot 3}
''')
    st.latex(r'''
A_3 = 1 \left(1 + 0.05\right)^3
''')
    st.latex(r'''
A_3 = 1 \cdot 1.157625
''')
    st.latex(r'''
A_3 = 1.157625
''')

    st.write("""
### Summary
- After 1 year, the investment grows to $1.05.
- After 2 years, the investment grows to approximately $1.10.
- After 3 years, the investment grows to approximately $1.16.

The principle of compound interest demonstrates how the initial investment grows exponentially over time, as each period's interest is calculated on the new total rather than just the principal. This is a fundamental concept in understanding long-term investment growth.
    """)

    st.write("""
#### 2. Market Returns on Different Asset Classes

Investments in various asset classes, such as stocks, bonds, and treasury bills, have historically yielded different rates of return. Understanding these returns is crucial for making informed investment decisions.

**a. Stocks**

Stocks represent ownership in companies and typically provide the highest returns over the long term. Stock values increase as companies grow and become more profitable. However, stocks are also highly volatile and can experience significant short-term fluctuations.

**b. Bonds**

Bonds are loans made to corporations or governments. They offer lower returns than stocks but are generally considered to be less risky. Intermediate-term bonds, which mature in 5-10 years, provide a steady stream of interest payments and moderate returns.

**c. Treasury Bills (T-Bills)**

Treasury bills are short-term government securities that mature in one year or less. They are considered very safe investments and offer lower returns compared to stocks and bonds. T-bills are often used as a benchmark for risk-free returns.

**d. Inflation**

Inflation measures the rate at which the general level of prices for goods and services rises, eroding purchasing power. To maintain their real value, investments must outperform inflation over time.

#### 3. Historical Performance of Investments

Historical data, such as the SBBI (Stocks, Bonds, Bills, and Inflation) indices, track the performance of different asset classes over extended periods. By analyzing this data, investors can understand how a $1 investment has grown over time in various assets:

**a. Small Company Stocks**

Small company stocks often have higher growth potential but come with higher risk. Historically, they have provided substantial long-term returns due to their growth prospects.

**b. Large Company Stocks (S&P 500)**

Large company stocks represent well-established companies with proven track records. These stocks provide steady returns with moderate risk, making them a popular choice for long-term investments.

**c. Intermediate-Term Bonds**

Intermediate-term bonds offer moderate returns with lower risk compared to stocks. They provide a stable income through regular interest payments and are less volatile than stocks.

**d. 30-Day T-Bills**

30-day T-bills are among the safest investments, offering the lowest returns. They are typically used by investors seeking to preserve capital and earn a modest return with minimal risk.

**e. Inflation-Adjusted Returns**

Adjusting for inflation is crucial for understanding the real growth of an investment. Investments need to outperform inflation to maintain their purchasing power over time.

#### 4. Practical Application of Historical Data

Understanding the growth of $1 investments over time helps investors in several ways:

**a. Investment Planning**

By knowing the historical performance of different asset classes, investors can make informed decisions about where to allocate their funds based on their risk tolerance and investment horizon.

**b. Risk Management**

Diversifying investments across various asset classes helps manage risk and ensures more stable returns over time. It mitigates the impact of poor performance in any single asset class.

**c. Inflation Protection**

Investing in assets that outperform inflation is essential to preserve and grow the real value of money. This ensures that investments retain their purchasing power over time.

#### 5. Visualization of Investment Growth

Visualizing the growth of $1 over time using historical data allows investors to see the effects of compounding and the differences in performance between various asset classes. It highlights the importance of long-term investing and the power of compound growth.

#### 6. Conclusion

Examining the growth of $1 investments over time provides valuable insights into the principles of compound interest, the historical performance of different asset classes, and strategies for effective investment planning and risk management. This knowledge is crucial for making informed decisions that can lead to wealth accumulation and financial stability over the long term.
""")

def historical_sharpe_ratio():
    st.title('Historical Sharpe Ratio of S&P 500')

    # Sidebar for user input
    st.sidebar.header('Input Parameters')
    start_year = st.sidebar.number_input('Start Year', min_value=1954, max_value=2023, value=1954)
    end_year = st.sidebar.number_input('End Year', min_value=1954, max_value=2023, value=2023)

    # Fetch S&P 500 data from Yahoo Finance
    sp500 = yf.download('^GSPC', start=f'{start_year}-01-01', end=f'{end_year}-12-31', progress=False)
    sp500['Monthly Return'] = sp500['Adj Close'].pct_change()

    # Fetch 3-month Treasury bill data from FRED
    fred = Fred(api_key=DEFAULT_API_KEY)
    tbill_series_id = 'DTB3'  # 3-Month Treasury Bill rate
    tbill_data = fred.get_series(tbill_series_id, start=f'{start_year}-01-01', end=f'{end_year}-12-31')
    tbill_df = pd.DataFrame(tbill_data, columns=['Rate'])
    tbill_df.index = pd.to_datetime(tbill_df.index)
    tbill_df = tbill_df.resample('M').ffill() / 100  # Convert percentage to decimal

    # Align S&P 500 and T-Bills data
    sp500_monthly = sp500['Monthly Return'].dropna()
    tbill_monthly = tbill_df['Rate'].reindex(sp500_monthly.index).fillna(method='ffill') / 12  # Monthly risk-free rate

    # Calculate monthly excess returns
    excess_returns = sp500_monthly - tbill_monthly

    # Calculate annualized return and standard deviation for each year
    annual_sharpe_ratios = {}
    for year in range(start_year, end_year + 1):
        annual_data = excess_returns[excess_returns.index.year == year]
        if len(annual_data) > 0:
            annual_return = (1 + annual_data).prod() ** (12 / len(annual_data)) - 1
            annual_std = annual_data.std() * np.sqrt(12)
            sharpe_ratio = annual_return / annual_std
            annual_sharpe_ratios[year] = sharpe_ratio

    # Convert to DataFrame for plotting
    sharpe_ratios_df = pd.DataFrame.from_dict(annual_sharpe_ratios, orient='index', columns=['Sharpe Ratio'])

    # Plot the Sharpe ratio
    plt.figure(figsize=(14, 7))
    plt.plot(sharpe_ratios_df.index, sharpe_ratios_df['Sharpe Ratio'], marker='o', linestyle='-')
    plt.title('Historical Sharpe Ratio of S&P 500')
    plt.xlabel('Year')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(range(start_year, end_year + 1, 5))
    plt.grid(True)
    st.pyplot(plt)

    st.write("""
    **Instructions:**
    1. Use the sidebar to select the start and end years for the data.
    2. The Sharpe Ratio is calculated as the ratio of excess returns (S&P 500 returns minus T-Bills rate) to the standard deviation of excess returns.
    3. The plot shows how the Sharpe Ratio of the S&P 500 has varied over the selected period.
    """)

    st.write("""
    **What is the Sharpe Ratio?**

    The Sharpe Ratio is a measure used to evaluate the risk-adjusted return of an investment or a portfolio. Developed by Nobel laureate William F. Sharpe, it helps investors understand the return of an investment compared to its risk. A higher Sharpe Ratio indicates better risk-adjusted performance.

    **Formula for Sharpe Ratio:**
    """)
    st.latex(r"""
    \text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
    """)
    st.write("""
    Where:
    - Rp is the return of the portfolio (or investment).
    - Rf is the risk-free rate of return.
    - sigma p is the standard deviation of the portfolio's excess return (i.e., \(R_p - R_f\)).

    **Components of the Sharpe Ratio:**
    - **Portfolio Return (Rp)**: The return of the investment or portfolio. In this case, it is the annual return of the S&P 500 index.
    - **Risk-Free Rate (Rf)**: The return on an investment with zero risk, typically represented by the yield on government Treasury bills. For the historical Sharpe Ratio, we use the 3-month Treasury bill rate.
    - **Standard Deviation of Excess Return (sigma p)**: Measures the volatility of the investment's excess return. This represents the investment's risk.

    **Historical Sharpe Ratio Calculation Steps:**
    1. **Fetch S&P 500 Data**: Download the historical adjusted close prices for the S&P 500 index from Yahoo Finance. Calculate the annual returns by resampling the daily adjusted close prices to yearly data and computing the percentage change.
    2. **Fetch Risk-Free Rate Data**: Retrieve the 3-month Treasury bill rate data from the Federal Reserve Economic Data (FRED) database. Resample the daily Treasury bill rates to yearly data and convert the rates to decimals.
    3. **Align Data**: Align the annual S&P 500 returns and the annualized Treasury bill rates to ensure they match for the same periods.
    4. **Calculate Excess Returns**: Compute the excess returns by subtracting the risk-free rate from the annual S&P 500 returns.
    5. **Calculate Sharpe Ratio**: Divide the excess returns by the standard deviation of the excess returns to obtain the Sharpe Ratio for each year.
    6. **Plot Sharpe Ratio**: Visualize the historical Sharpe Ratio over the selected period.
    """)


def app():
    st.title('Portfolio Theory Page')

    if "page" not in st.session_state:
        st.session_state.page = "Portfolio Theory Description"

    page = st.sidebar.selectbox("Select a page", ["Portfolio Theory Description", "Inflation vs T-Bills", "Inflation vs T-Bills API", "Small vs Large Portfolio", "Small vs Large Portfolio API", "Annual T-bill returns", "Annual T-bill returns API", "Annual S&P 500 returns API", "Annualized SD of SP500","Growth of a dollar","Historical Sharpe Ratio" ])
    st.session_state.page = page

    if st.session_state.page == "Portfolio Theory Description":
        portfolio_description()
    elif st.session_state.page == "Inflation vs T-Bills":
        inflation_vs_tbills()
    elif st.session_state.page == "Inflation vs T-Bills API":
        inflation_vs_tbills_api()
    elif st.session_state.page == "Small vs Large Portfolio":
        small_vs_large_portfolio()
    elif st.session_state.page == "Small vs Large Portfolio API":
        small_vs_large_portfolio_api()
    elif st.session_state.page == "Annual T-bill returns":
        Annual_tbill_returns()
    elif st.session_state.page == "Annual T-bill returns API":
        Annual_tbill_returns_API()
    elif st.session_state.page == "Annual S&P 500 returns API":
        Annual_SP500_returns_API()
    elif st.session_state.page == "Annualized SD of SP500":
        Annualized_Std_Dev_Sp500()
    elif st.session_state.page == "Growth of a dollar":
        growth_of_dollar()
    elif st.session_state.page == "Historical Sharpe Ratio":
        historical_sharpe_ratio()

if __name__ == "__main__":
    app()
