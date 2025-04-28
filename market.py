import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def market_efficiency_definition():
    # Title
    st.title("Understanding Market Efficiency")

    # Introduction
    st.markdown(
        """
        Market efficiency refers to the extent to which market prices reflect all available information.
        According to the **Efficient Market Hypothesis (EMH)**, it is impossible to consistently achieve returns
        higher than average market returns on a risk-adjusted basis, as prices always incorporate and reflect all relevant information.
        """
    )

    # Types of Market Efficiency
    st.header("Types of Market Efficiency")

    st.subheader("1. Weak-Form Efficiency")
    st.markdown(
        """
        - In a weak-form efficient market, current stock prices reflect all **past trading information** (e.g., price and volume data).
        - Technical analysis is ineffective as past price patterns cannot predict future prices.
        - However, fundamental analysis may still provide opportunities for excess returns.
        """
    )

    st.subheader("2. Semi-Strong Form Efficiency")
    st.markdown(
        """
        - In a semi-strong form efficient market, prices incorporate **all publicly available information**, including financial statements, news, and economic data.
        - Neither technical analysis nor fundamental analysis can provide an edge, as all available information is already priced in.
        - Only insider information can potentially lead to abnormal profits.
        """
    )

    st.subheader("3. Strong-Form Efficiency")
    st.markdown(
        """
        - In a strong-form efficient market, prices reflect **all information, both public and private (insider information).**
        - Even insider trading would not provide an advantage in such a market.
        - Empirical evidence suggests that markets are not perfectly strong-form efficient.
        """
    )

    # Implications of Market Efficiency
    st.header("Implications of Market Efficiency")
    st.markdown(
        """
        - If markets are efficient, it is difficult for investors to consistently outperform the market.
        - **Passive investing**, such as index fund investing, is a preferred strategy in highly efficient markets.
        - Market anomalies (e.g., momentum effects, January effect) suggest that markets may not be perfectly efficient.
        """
    )

def task():
    st.title("Task - Analyze Market Efficiency")

    # Displaying the first image
    st.write("#### Graph 1:")
    st.image("market_efficiency_task_images/walgreens.png")

    st.write("### Question:")
    st.text_area("What form of market efficiency applies to the drop in Walgreens' stock price due to weak consumer demand?", key="efficiency_answer1", height=140)

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
    st.text_area("What form of market efficiency applies to the surge in Reddit's stock price due to strong earnings?", key="efficiency_answer2", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1.	https://www.bloomberg.com/news/articles/2024-10-29/reddit-signals-strong-holiday-quarter-to-come-shares-soar
    2.	https://www.wsj.com/livecoverage/stock-market-today-earnings-dow-sp500-nasdaq-live-10-29-2024/card/reddit-turns-a-profit-grows-revenue-and-users-BfRUp5yZrbtjLVqOydRU
    """)

    # Displaying the third image
    st.write("#### Graph 3:")
    st.image("market_efficiency_task_images/AppLovin.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What form of market efficiency applies to the surge in AppLovin's stock price due to an upgrade and strong earnings?", key="efficiency_answer3", height=140)

    st.write("#### Citations:")
    st.markdown("""
    1. https://www.nasdaq.com/articles/daiwa-capital-upgrades-applovin-app
    2. https://www.investopedia.com/dow-jones-today-11072024-8741258
    """)

    # Displaying the fourth image
    st.write("#### Graph 4:")
    st.image("market_efficiency_task_images/Tesla.png")  # You can use another image here if needed

    st.write("### Question:")
    st.text_area("What form of market efficiency applies to the surge in Tesla's stock price due to an election victory and Elon Musk's involvement?", key="efficiency_answer4", height=140)

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
    st.text_area("What form of market efficiency applies to the drop in CrowdStrike's stock price due to a software outage?", key="efficiency_answer5", height=140)

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
    st.text_area("What form of market efficiency applies to the drop in Nvidia's stock price, resulting in a $600 billion market cap loss?", key="efficiency_answer6", height=140)

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
