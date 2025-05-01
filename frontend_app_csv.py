import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------
@st.cache_data
def load_base_currencies():
    url = "https://raw.githubusercontent.com/elvin1212/currency-data/main/USD_from_1999_cleaned.csv"
    df = pd.read_csv(url)
    return sorted(df['base_currency'].unique()) if 'base_currency' in df.columns else ['USD', 'EUR', 'JPY', 'GBP', 'AUD']

@st.cache_data
def load_date_range(base_currency):
    url = f"https://raw.githubusercontent.com/elvin1212/currency-data/main/{base_currency}_from_1999_cleaned.csv"
    df = pd.read_csv(url, parse_dates=["date"])
    return df["date"].min(), df["date"].max()

@st.cache_data
def load_filtered_data(base_currency, selected_currencies, start_date, end_date):
    url = f"https://raw.githubusercontent.com/elvin1212/currency-data/main/{base_currency}_from_1999_cleaned.csv"
    df = pd.read_csv(url, parse_dates=["date"])
    df = df[df["quote_currency"].isin(selected_currencies)]
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    return df

def simulate_portfolio(df, selected_currencies, weights):
    df_pivot = df.pivot(index='date', columns='quote_currency', values='rate').dropna()
    df_pivot = df_pivot[selected_currencies]

    daily_returns = df_pivot.pct_change().dropna()
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod()

    annualized_return = (portfolio_value.iloc[-1]) ** (1 / (len(portfolio_value) / 252)) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    drawdown = (portfolio_value / portfolio_value.cummax()) - 1
    max_drawdown = drawdown.min()

    return {
        "portfolio_value": portfolio_value,
        "portfolio_returns": portfolio_returns,
        "drawdown": drawdown,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }

# ---------------------
st.set_page_config(page_title="Currency Portfolio Simulator", layout="wide")
st.sidebar.header("Simulation Settings")

base_currency = st.sidebar.selectbox("Select Base Currency", load_base_currencies())

try:
    min_date, max_date = load_date_range(base_currency)
    url = f"https://raw.githubusercontent.com/elvin1212/currency-data/main/{base_currency}_from_1999_cleaned.csv"
    df_temp = pd.read_csv(url)
    quote_currencies = sorted(df_temp['quote_currency'].unique().tolist())
except Exception as e:
    st.error("⚠️ Error loading initial data. Please check data file or URL format.")
    st.stop()

selected_currencies = st.sidebar.multiselect("Select currencies to invest", quote_currencies)

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

weights = []
if selected_currencies:
    st.sidebar.markdown("### Set Investment Weights")
    for currency in selected_currencies:
        weight = st.sidebar.number_input(f"Weight for {currency}", min_value=0.0, max_value=1.0, step=0.01, value=round(1/len(selected_currencies), 2))
        weights.append(weight)

if weights and sum(weights) > 0:
    weights = np.array(weights)
    weights = weights / weights.sum()

st.title("Currency Portfolio Backtest Dashboard")

if selected_currencies and weights is not None and sum(weights) > 0:
    df = load_filtered_data(base_currency, selected_currencies, start_date, end_date)

    if not df.empty:
        result = simulate_portfolio(df, selected_currencies, weights)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annualized Return", f"{result['annualized_return']:.2%}")
        col2.metric("Annualized Volatility", f"{result['annualized_volatility']:.2%}")
        col3.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
        col4.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")

        st.subheader("Portfolio Cumulative Return Over Time")
        fig1 = px.line(result['portfolio_value'], labels={"value": "Portfolio Value", "date": "Date"})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Portfolio Daily Return Histogram")
        fig2 = px.histogram(result['portfolio_returns'], nbins=50, labels={"value": "Daily Return"}, title="Distribution of Daily Returns")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Portfolio Drawdown Over Time")
        fig3 = px.line(result['drawdown'], labels={"value": "Drawdown", "date": "Date"})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No data found for the selected parameters. Please adjust your selection.")
else:
    st.info("Please select at least one currency and specify investment weights.")
