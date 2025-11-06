from pathlib import Path
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from api.polygon.client import PolygonTradesDriver

alt.data_transformers.disable_max_rows()
TZ_NY = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")


# --------------------------- UI & App ---------------------------------
def main() -> None:
    st.set_page_config(page_title="Market Reply", layout="wide")

    st.title("Market Reply")
    st.caption("Trades + Bid/Ask snapshot around a selected NY time")

    with st.form("controls"):
        c1, c2 = st.columns(2)
        ticker = c1.text_input("Ticker", key="ticker", placeholder="AAAPL, etc.").strip().upper()

        period_min = c2.number_input(
            "Window (minutes)",
            min_value=1, max_value=120, value=10, step=5,
            help="Shows ± this many minutes around the selected time.",
            key="period",
        )

        c3, c4 = st.columns(2)
        the_date = c3.date_input("Date", value=date.today(), format="YYYY-MM-DD", key="date")

        # time_input requires step >= 60 seconds
        c4.text_input("NY Time (`HH:MM:SS`))", key='time', value='09:00:00', help="24h format, e.g., 09:30:05")
        time_str = st.session_state.get("time")

        api_key = st.text_input("Polygon API Key", type="password", key="key")

        # MUST be inside the form:
        run = st.form_submit_button("Submit")

    if not run:
        return

    if not ticker:
        st.error("Please enter a ticker.")
        return
    if not api_key:
        st.error("Please enter your Polygon API key.")
        return

    # Convert NY datetime to UTC
    hh, mm, ss = map(int, time_str.split(":"))
    tz_ny = ZoneInfo("America/New_York")
    local_dt = datetime(the_date.year, the_date.month, the_date.day, hh, mm, ss, tzinfo=tz_ny)

    # 4) Convert to UTC
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))
    time_utc = utc_dt.strftime('%Y-%m-%d %H:%M:%S')

    try:
        start_utc, end_utc = shift_window(time_utc, period_min)
        df_trades, df_quotes, df_exchanges = fetch_data(api_key, ticker, start_utc, end_utc)

        if df_trades.empty and df_quotes.empty:
            st.warning("No trades or quotes returned for the selected window.")
            return

        trade_quote_df = prepare_joined_df(df_trades, df_quotes, df_exchanges)
        chart = build_chart(trade_quote_df, ticker)
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.exception(e)


# --------------------------- Data layer ---------------------------------
@st.cache_data(show_spinner=False)
def fetch_exchanges_cached() -> pd.DataFrame:

    df_exchanges = pd.read_parquet('data/exchanges.parquet', engine='pyarrow')
    df_exchanges = df_exchanges.set_index('id')[['name', 'mic']]

    return df_exchanges


def fetch_data(api_key: str, ticker: str, start_utc: str, end_utc: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull trades, quotes, and exchanges for a window."""
    with PolygonTradesDriver(api_key=api_key) as drv:
        trades = drv.get_trades(ticker, start_utc, end_utc, limit=10000)
        quotes = drv.get_quotes(ticker, start_utc, end_utc, limit=10000)

    df_trades = pd.DataFrame(trades)
    df_quotes = pd.DataFrame(quotes)

    # Timestamps to datetime (ns → pandas datetime)
    for df, cols in [(df_trades, ["participant_timestamp", "sip_timestamp"]),
                     (df_quotes, ["participant_timestamp", "sip_timestamp"])]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="ns", errors="coerce")

    df_exchanges = fetch_exchanges_cached()

    return df_trades, df_quotes, df_exchanges


def prepare_joined_df(df_trades: pd.DataFrame, df_quotes: pd.DataFrame, df_exchanges: pd.DataFrame) -> pd.DataFrame:
    """Clean columns, map exchange IDs → MICs, join trades & quotes on participant time, ffill quotes."""
    # --- Trades
    trade_cols = {
        "participant_timestamp": "time",
        "price": "trade_price",
        "size": "trade_size",
        "exchange": "trade_ex",
    }
    trades_needed = [c for c in trade_cols.keys() if c in df_trades.columns]
    df_trades_cleaned = (
        df_trades[trades_needed]
        .rename(columns=trade_cols)
        .sort_values("time", kind="stable")
    )

    # --- Quotes
    quote_cols = ["ask_price", "ask_size", "ask_exchange", "bid_price", "bid_size", "bid_exchange"]
    quotes_needed = ["participant_timestamp"] + [c for c in quote_cols if c in df_quotes.columns]
    df_quotes_cleaned = (
        df_quotes[quotes_needed]
        .rename(columns={"participant_timestamp": "time"})
        .sort_values("time", kind="stable")
    )

    # --- Join
    trade_quote_df = pd.merge(df_trades_cleaned, df_quotes_cleaned, on='time', how='outer')

    lookup = df_exchanges['mic'].to_dict()
    ex_cols = ['trade_ex', 'ask_exchange', 'bid_exchange']
    quotes_col = ['ask_price', 'ask_size', 'ask_exchange', 'bid_price', 'bid_size', 'bid_exchange']
    trade_quote_df[ex_cols] = trade_quote_df[ex_cols].replace(lookup)

    trade_quote_df[quotes_col] = trade_quote_df[quotes_col].ffill()

    return trade_quote_df


# --------------------------- Charting -----------------------------------
def build_chart(trade_quote_df: pd.DataFrame, ticker: str) -> alt.Chart:
    """Create layered Altair chart: bid/ask band, trades, and quote points."""
    # Y domain: gentle band around mid (robust to outliers)
    prices = pd.concat(
        [
            trade_quote_df.get("trade_price", pd.Series(dtype=float)),
            trade_quote_df.get("bid_price", pd.Series(dtype=float)),
            trade_quote_df.get("ask_price", pd.Series(dtype=float)),
        ],
        ignore_index=True,
    )
    y_min = prices.mean() * 0.995
    y_max = prices.mean() * 1.005

    base = alt.Chart(trade_quote_df).encode(
        x=alt.X("time:T", title="Time")
    )

    q_area = base.mark_area(opacity=0.2, color="#fbf48e").encode(
        alt.Y("bid_price:Q", title='Bid Price', scale=alt.Scale(domain=[y_min, y_max])),
        alt.Y2("ask_price:Q", title='Ask Price')
    )

    t_point = base.mark_circle(opacity=1, filled=True).encode(
        alt.Y("trade_price:Q", title="Trade Price"),
        alt.Size('trade_size', title='Trade Size', scale=alt.Scale(range=[20, 500])),
        alt.Color('trade_ex:N', title='Exchange', scale=alt.Scale(scheme='category10')),
        tooltip=['trade_price', 'trade_size', 'trade_ex']
    )

    q_area_a = base.mark_point(color="#006c09").encode(
        alt.Y("ask_price:Q", title="Ask Price"),
        alt.Size('ask_size', title='Trade Size', scale=alt.Scale(range=[2, 5])),
        tooltip=['ask_price', 'ask_size', 'ask_exchange']
    )

    q_area_b = base.mark_point(color="#7b0000").encode(
        alt.Y("bid_price:Q", title="Bid Price"),
        alt.Size('bid_size', title='Trade Size', scale=alt.Scale(range=[2, 5])),
        tooltip=['bid_price', 'bid_size', 'bid_exchange']
    )

    bid_line = base.mark_line(opacity=0.7, strokeWidth=1, color="#7b0000").encode(y="bid_price:Q")
    ask_line = base.mark_line(opacity=0.7, strokeWidth=1, color="#006c09").encode(y="ask_price:Q")

    chart = (q_area + t_point + bid_line + ask_line + q_area_a + q_area_b).properties(
        width=900,
        height=450,
        title=f"{ticker} – Trades and Bid-Ask Spread"
    ).resolve_scale(
        size='independent'
    ).interactive()

    return chart


# --------------------------- Utilities -----------------------------------
def shift_window(dt_str_utc: str, period_m: int) -> tuple[str, str]:
    """Return (start_utc_str, end_utc_str) surrounding dt_str_utc by ±period_m."""
    dt = datetime.strptime(dt_str_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_UTC)
    start = (dt - timedelta(minutes=period_m)).strftime("%Y-%m-%d %H:%M:%S")
    end = (dt + timedelta(minutes=period_m)).strftime("%Y-%m-%d %H:%M:%S")
    return start, end


if __name__ == "__main__":
    main()
