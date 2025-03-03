import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

def market_efficiency_definition():
    st.write("Market efficiency is a concept in financial economics that suggests that financial markets reflect all available information...")

def task():
    st.title("Task - Analyze Market Efficiency")

    # Displaying the first image
    st.write("#### Graph 1:")
    st.image("market_efficiency_task_images/walgreens.png")

    st.write("### Question:")
    st.text_area("What kind of market efficiency is this?", key="efficiency_answer1", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1. https://www.investopedia.com/walgreens-stock-sinks-to-27-year-lows-amid-weak-consumer-demand-8670428
    2. https://www.wsj.com/livecoverage/stock-market-today-dow-sp500-nasdaq-live-06-27-2024/card/walgreens-stock-tests-lowest-levels-in-decades-DZaKcTEkPkd99esb2aO0
    3. https://www.cnbc.com/2024/06/27/walgreens-wba-earnings-q3-2024.html
    4. https://www.aaii.com/investingideas/article/216962-why-walgreens-boots-alliance-inc8217s-wba-stock-is-down-2216
    5. https://investor.walgreensbootsalliance.com/news-releases/news-release-details/walgreens-boots-alliance-reports-fiscal-2024-third-quarter
    6. https://www.wsj.com/livecoverage/stock-market-today-dow-sp500-nasdaq-live-06-27-2024
    7. https://www.reuters.com/business/retail-consumer/walgreens-cuts-2024-profit-forecast-announces-store-closures-2024-06-27/
    8. https://www.forbes.com/sites/tylerroush/2024/10/15/walgreens-closing-1200-stores-as-earnings-beat-projections/
    """)

    # Displaying the second image
    st.write("#### Graph 2:")
    st.image("market_efficiency_task_images/Reddit.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What kind of market efficiency is this?", key="efficiency_answer2", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1.	https://www.bloomberg.com/news/articles/2024-10-29/reddit-signals-strong-holiday-quarter-to-come-shares-soar
    2.	https://www.wsj.com/livecoverage/stock-market-today-earnings-dow-sp500-nasdaq-live-10-29-2024/card/reddit-turns-a-profit-grows-revenue-and-users-BfRUp5yZrbtjLVqOydRU
    """)

    # Displaying the third image
    st.write("#### Graph 3:")
    st.image("market_efficiency_task_images/AppLovin.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What kind of market efficiency is this?", key="efficiency_answer3", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1. https://www.nasdaq.com/articles/daiwa-capital-upgrades-applovin-app
    2. https://www.investopedia.com/dow-jones-today-11072024-8741258
    """)

    # Displaying the fourth image
    st.write("#### Graph 4:")
    st.image("market_efficiency_task_images/Tesla.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What kind of market efficiency is this?", key="efficiency_answer4", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1. https://esgnews.com/tesla-shares-soar-12-percent-in-premarket-trading-following-trump-reelection/
    2. https://nypost.com/2024/11/06/business/tesla-shares-surge-on-trump-victory-after-musks-campaign-support/
    3. https://www.wsj.com/livecoverage/stock-market-today-fed-meeting-dow-nasdaq-sp500-live-11-06-2024/card/tesla-stock-soars-premarket-bucking-wider-ev-selloff-WQ602tHDJHsL1oO0GeJF
    4. https://www.nytimes.com/2024/11/06/business/tesla-stock-elon-musk-trump.html
    5. https://www.bloomberg.com/news/articles/2024-11-06/tesla-soars-as-musk-s-all-in-bet-on-trump-seen-reaping-rewards?embedded-checkout=true
    """)

     # Displaying the fifth image
    st.write("#### Graph 5:")
    st.image("market_efficiency_task_images/crowdstrike.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What kind of market efficiency is this?", key="efficiency_answer5", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1. https://roboforex.com/beginners/analytics/forex-forecast/stocks/stocks-forecast-crwd-2024/
    2. https://www.morningstar.com/markets/crowdstrike-share-fall-offers-buying-opportunity
    3. https://www.forbes.com/sites/petercohan/2024/07/19/crowdstrike-stock-falls-post-outage-but-there-are-still-reasons-to-buy/
    """)

     # Displaying the sixth image
    st.write("#### Graph 6:")
    st.image("market_efficiency_task_images/Nvidia.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What kind of market efficiency is this?", key="efficiency_answer6", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1. https://www.cnbc.com/2025/01/27/nvidia-sheds-almost-600-billion-in-market-cap-biggest-drop-ever.html
    2. https://www.investopedia.com/dow-jones-today-01272025-8780724
    3. https://fortune.com/2025/01/27/nvidia-deepseek-rout-tech-stocks/
    """)

def app():
    page = st.sidebar.selectbox("Select Page", ["Market Efficiency Description", "Task"])

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
