import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration for a cleaner look
st.set_page_config(page_title="Stock Data Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {width: 100%; padding: 10px; font-size: 16px;}
    .stTextInput>div>div>input {font-size: 16px;}
    .stDateInput>div>div>input {font-size: 16px;}
    h1, h2, h3 {color: #1f2a44;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.header("Stock Data Dashboard")
    st.markdown("Enter a stock ticker and select a date range to view the data.")

    # Stock ticker input with placeholder and help text
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, GOOGL, MSFT",
        help="Enter a valid stock ticker symbol (e.g., AAPL for Apple)."
    )

    # Date range inputs with defaults
    default_end = datetime.today()
    default_start = default_end - timedelta(days=30)
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=datetime(2000, 1, 1),
        max_value=default_end,
        help="Select the start date for the data."
    )
    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
        max_value=default_end,
        help="Select the end date for the data."
    )

    # Button to fetch data
    fetch_button = st.button("Get Stock Data")

# Main content area
st.title("Stock Data Dashboard")
st.markdown("Explore stock price trends, key metrics, and raw data for your selected stock.")

# Initialize session state for data
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Fetch and display data when button is clicked
if fetch_button:
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        st.session_state.stock_data = stock.history(start=start_date, end=end_date)

        # Fetch stock info
        stock_info = stock.info

        # Display error if no data is found
        if st.session_state.stock_data.empty:
            st.error("No data found for the given ticker or date range. Please check your inputs.")
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}. Try a valid ticker like AAPL or GOOGL.")

# Tabs for organizing content
if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
    tab1, tab2, tab3 = st.tabs(["📈 Chart", "ℹ️ Stock Info", "📊 Raw Data"])

    with tab1:
        st.subheader(f"{ticker.upper()} Stock Price")
        # Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=st.session_state.stock_data.index,
                open=st.session_state.stock_data['Open'],
                high=st.session_state.stock_data['High'],
                low=st.session_state.stock_data['Low'],
                close=st.session_state.stock_data['Close'],
                name=ticker.upper()
            )
        ])
        fig.update_layout(
            title=f"{ticker.upper()} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"{ticker.upper()} Information")
        stock_info = yf.Ticker(ticker).info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Company", stock_info.get('longName', 'N/A'))
            st.metric("Sector", stock_info.get('sector', 'N/A'))
            st.metric("Market Cap", f"${stock_info.get('marketCap', 'N/A'):,}")
        with col2:
            st.metric("P/E Ratio", stock_info.get('trailingPE', 'N/A'))
            st.metric("52-Week High", f"${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.metric("52-Week Low", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')}")

    with tab3:
        st.subheader("Raw Data")
        st.dataframe(st.session_state.stock_data, use_container_width=True)
else:
    st.info("Enter a stock ticker and date range in the sidebar, then click 'Get Stock Data' to view results.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data provided by Yahoo Finance")
