import streamlit as st
st.set_page_config(page_title="Spark-Enhanced Portfolio Simulator", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
from pymongo import MongoClient
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import *

@st.cache_resource
def get_pg_engine():
    return create_engine('postgresql://postgres:123@localhost:5434/5400_final')

@st.cache_resource
def get_mongo_collection():
    client = MongoClient("mongodb://localhost:27017")
    db = client["portfolio_simulation"]
    return db["user_inputs"]

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("CurrencyBacktestApp").getOrCreate()

pg_engine = get_pg_engine()
mongo_col = get_mongo_collection()
spark = get_spark_session()

def load_base_currencies():
    query = "SELECT DISTINCT base_currency FROM exchange_rate ORDER BY base_currency;"
    return pd.read_sql(query, pg_engine)['base_currency'].tolist()

def load_date_range(base_currency):
    query = f"""
    SELECT MIN(date) AS min_date, MAX(date) AS max_date
    FROM exchange_rate
    WHERE base_currency = '{base_currency}'
    """
    df = pd.read_sql(query, pg_engine)
    return df.iloc[0]['min_date'], df.iloc[0]['max_date']

def simulate_portfolio_core(df_rates, quote_currencies, weights):
    df_pivot = df_rates.pivot(index='date', columns='quote_currency', values='rate').dropna()
    df_pivot = df_pivot[quote_currencies]
    daily_returns = df_pivot.pct_change().dropna()
    weights = np.array(weights)
    weights /= weights.sum()
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

def simulate_batch_backtest(pdf_iter):
    pg_engine = create_engine("postgresql://postgres:123@localhost:5434/5400_final")
    for row in pdf_iter:
        base = row["base_currency"].iloc[0]
        quotes = row["quote_currencies"].iloc[0]
        weights = row["weights"].iloc[0]
        start = row["start_date"].iloc[0]
        end = row["end_date"].iloc[0]
        sql = f"""
        SELECT date, quote_currency, rate
        FROM exchange_rate
        WHERE base_currency = '{base}'
          AND quote_currency IN ({','.join(f"'{q}'" for q in quotes)})
          AND date BETWEEN '{start}' AND '{end}'
        """
        df_rates = pd.read_sql(sql, pg_engine)
        df_rates["date"] = pd.to_datetime(df_rates["date"])
        result = simulate_portfolio_core(df_rates, quotes, weights)
        yield pd.DataFrame([{
            "user_id": row["user_id"].iloc[0],
            "annualized_return": result["annualized_return"],
            "annualized_volatility": result["annualized_volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "portfolio_value": result["portfolio_value"].to_json(date_format='iso'),
            "portfolio_returns": result["portfolio_returns"].to_json(date_format='iso'),
            "drawdown": result["drawdown"].to_json(date_format='iso')
        }])

# === Streamlit UI ===
st.sidebar.header("Portfolio Input")
user_id = st.sidebar.text_input("User ID")
base_currency = st.sidebar.selectbox("Base Currency", load_base_currencies())
min_date, max_date = load_date_range(base_currency)

query = f"SELECT DISTINCT quote_currency FROM exchange_rate WHERE base_currency = '{base_currency}'"
quote_currencies_all = pd.read_sql(query, pg_engine)['quote_currency'].tolist()
quote_currencies = st.sidebar.multiselect("Quote Currencies", quote_currencies_all)

# 定义联动函数
def update_slider(c):
    st.session_state[f"{c}_slider"] = st.session_state[f"{c}_num"]

def update_num(c):
    st.session_state[f"{c}_num"] = st.session_state[f"{c}_slider"]

# 构建控件和同步
weights = []
for c in quote_currencies:
    st.session_state.setdefault(f"{c}_num", 0.0)
    st.session_state.setdefault(f"{c}_slider", 0.0)

    col1, col2 = st.sidebar.columns([2, 3])
    with col1:
        st.number_input(
            f"Weight ({c})", min_value=0.0, max_value=1.0, step=0.05,
            key=f"{c}_num", on_change=update_slider, args=(c,)
        )
    with col2:
        st.slider(
            " ", min_value=0.0, max_value=1.0, step=0.05,
            key=f"{c}_slider", on_change=update_num, args=(c,)
        )

    weights.append(st.session_state[f"{c}_num"])


remaining_weight = 1.0 - sum(weights)
if remaining_weight < 0:
    weights = np.array(weights) / sum(weights)
    st.sidebar.warning("Total weight exceeded 1. Automatically normalized.")
else:
    st.sidebar.info(f"Remaining weight: {remaining_weight:.2f}")

start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

submit = st.sidebar.button("Run Simulation")

if submit and quote_currencies and user_id:
    # Step 1: 构造输入数据
    portfolio_df = pd.DataFrame([{
        "user_id": user_id,
        "base_currency": base_currency,
        "quote_currencies": quote_currencies,
        "weights": weights,
        "start_date": str(start_date),
        "end_date": str(end_date)
    }])
    df_portfolios = spark.createDataFrame(portfolio_df)

    # Step 2: Spark 模拟
    schema = StructType([
        StructField("user_id", StringType()),
        StructField("annualized_return", DoubleType()),
        StructField("annualized_volatility", DoubleType()),
        StructField("sharpe_ratio", DoubleType()),
        StructField("max_drawdown", DoubleType()),
        StructField("portfolio_value", StringType()),
        StructField("portfolio_returns", StringType()),
        StructField("drawdown", StringType())
    ])
    df_result = df_portfolios.mapInPandas(simulate_batch_backtest, schema=schema).toPandas()
    res = df_result.iloc[0]

    # Step 3: 展示指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{res['annualized_return']:.2%}")
    col2.metric("Annualized Volatility", f"{res['annualized_volatility']:.2%}")
    col3.metric("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
    col4.metric("Max Drawdown", f"{res['max_drawdown']:.2%}")

    # Step 4: 图表展示
    st.subheader("Portfolio Cumulative Return Over Time")
    df_cum = pd.read_json(res['portfolio_value'], typ='series').to_frame(name='Cumulative Return')
    fig1 = px.line(df_cum, x=df_cum.index, y='Cumulative Return')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Portfolio Daily Return Histogram")
    df_daily = pd.read_json(res['portfolio_returns'], typ='series').to_frame(name='Daily Return')
    fig2 = px.histogram(df_daily, x='Daily Return', nbins=50, title="Distribution of Daily Returns")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Portfolio Drawdown Over Time")
    df_draw = pd.read_json(res['drawdown'], typ='series').to_frame(name='Drawdown')
    fig3 = px.line(df_draw, x=df_draw.index, y='Drawdown')
    st.plotly_chart(fig3, use_container_width=True)

    # Step 5: 自动保存到 MongoDB
    mongo_col.insert_one({
        "user_id": user_id,
        "base_currency": base_currency,
        "portfolio": list(zip(quote_currencies, weights)),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "result": {
            "annualized_return": res['annualized_return'],
            "annualized_volatility": res['annualized_volatility'],
            "sharpe_ratio": res['sharpe_ratio'],
            "max_drawdown": res['max_drawdown']
        },
        "timestamp": datetime.utcnow()
    })
    st.success("✅ Simulation result saved to MongoDB.")

