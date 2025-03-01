import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Fetch stock prices function
def fetch_stock_prices(stock, event_date, days=50):
    event_datetime = datetime.strptime(event_date, "%Y-%m-%d")
    start_date = (event_datetime - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = (event_datetime + timedelta(days=days)).strftime("%Y-%m-%d")

    df = yf.download(stock, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data available for {stock}.")
        return pd.DataFrame()

    df['Days Relative'] = (df.index - event_datetime).days
    df['Price'] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']

    return df

# Plot stock price graph
def plot_stock_price(stock, df, event, event_date):
    event_datetime = datetime.strptime(event_date, "%Y-%m-%d")
    
    if event_datetime not in df.index:
        event_datetime = df.index.get_indexer([event_datetime], method='nearest')
        event_datetime = df.index[event_datetime[0]]
        st.warning(f"No data available on {event_date}, using nearest available date: {event_datetime}")

    plt.figure(figsize=(12, 6))
    plt.plot(df['Days Relative'], df['Price'], label=stock, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', label='Event Day')
    plt.scatter(0, df.loc[event_datetime, 'Price'], color='red', zorder=3, label=f"Event Date: {event_datetime}")

    plt.title(f"Stock Prices Around {event}")
    plt.xlabel("Days Relative to Event")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Display Question Section
def display_exam_question(stock, event_name, event_date, citations):
    st.subheader(f"{event_name} ({stock})")
    df = fetch_stock_prices(stock, event_date)
    if not df.empty:
        plot_stock_price(stock, df, event_name, event_date)

    st.write("### Question:")
    st.write("What kind of market efficiency is this situation?")
    user_answer = st.text_area(f"Your Answer for {stock}")

    # Citations
    st.write("#### Citations:")
    for citation in citations:
        st.markdown(f"- [{citation}]({citation})")

    return user_answer

def task():
    st.title("Market Efficiency Exam")

    answers = {}
    
    # Nvidia Example
    citations_nvidia = [
        "https://www.cnbc.com/2025/01/27/nvidia-sheds-almost-600-billion-in-market-cap-biggest-drop-ever.html",
        "https://www.investopedia.com/dow-jones-today-01272025-8780724",
        "https://fortune.com/2025/01/27/nvidia-deepseek-rout-tech-stocks/"
    ]
    answer_nvda = display_exam_question("NVDA", "Nvidia Shares Drop, Shedding $600 Billion in Market Cap", "2025-01-27", citations_nvidia)
    answers["Nvidia"] = answer_nvda

    if st.button("Submit Answers"):
        st.success("Your answers have been submitted!")
        st.json(answers)
