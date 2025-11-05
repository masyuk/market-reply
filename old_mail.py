import pandas as pd
import numpy as np
import altair as alt
import re
import streamlit as st
from datetime import date, time, datetime, timedelta
from api.polygon.client import PolygonTradesDriver
from zoneinfo import ZoneInfo
alt.data_transformers.disable_max_rows()

def main():
    st.set_page_config(page_title="Market Reply")

    col1, col2 = st.columns(2)
    ticker = col1.text_input("Enter your ticker", key="ticker")
    period = col2.number_input("Enter period (minutes)", min_value=1, max_value=120, value=10, step=5, key="period", help='+/- a few minutes from the selected time')

    col1.date_input("Date", value=date.today(), key='date', format="YYYY-MM-DD")

    default_time_str = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S")
    col2.text_input("NY Time (HH:MM:SS)", key='time', value=default_time_str, help="24h format, e.g., 09:30:05")

    password = st.text_input("Enter API Key", key="key")

    date_str = st.session_state.get("date")
    time_str = st.session_state.get("time")

    if not re.fullmatch(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$", time_str):
        st.error("Please enter time as HH:MM:SS (00–23:00–59:00–59).")
        st.stop()

    hh, mm, ss = map(int, time_str.split(":"))
    tz_ny = ZoneInfo("America/New_York")
    local_dt = datetime(date_str.year, date_str.month, date_str.day, hh, mm, ss, tzinfo=tz_ny)

    # 4) Convert to UTC
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))
    time_utc = utc_dt.strftime('%Y-%m-%d %H:%M:%S')

    if st.button("Run"):
        _run_after_enter(password, ticker, time_utc, period)

def _run_after_enter(key, ticker, time_utc, period_m):
    period = _shift_datetime(time_utc, period_m)

    with PolygonTradesDriver(api_key=key) as drv:
        trades = drv.get_trades(ticker, period[0], period[1], limit=10000)
        df_trades = pd.DataFrame(trades)

        qoutes = drv.get_quotes(ticker, period[0], period[1], limit=10000)
        df_quotes = pd.DataFrame(qoutes)

        exchanges = drv.get_exchanges()

    df_exchanges = pd.DataFrame(exchanges['results'])

    df_exchanges = df_exchanges.set_index('id')[['name', 'mic']]

    df_trades['participant_timestamp'] = pd.to_datetime(df_trades['participant_timestamp'], unit='ns')
    df_trades['sip_timestamp'] = pd.to_datetime(df_trades['sip_timestamp'], unit='ns')

    df_quotes['participant_timestamp'] = pd.to_datetime(df_quotes['participant_timestamp'], unit='ns')
    df_quotes['sip_timestamp'] = pd.to_datetime(df_quotes['sip_timestamp'], unit='ns')

    df_trades_cleaned = df_trades[['participant_timestamp', 'price', 'size', 'exchange']].rename(
        columns={
            'participant_timestamp': 'time',
            'price': 'trade_price',
            'size': 'trade_size',
            'exchange': 'trade_ex'}
            ).sort_values('time')

    quotes_col = ['ask_price', 'ask_size', 'ask_exchange', 'bid_price', 'bid_size', 'bid_exchange']
    df_quotes_cleaned = df_quotes[['participant_timestamp'] + quotes_col].rename(columns={'participant_timestamp': 'time'}).sort_values('time')

    trade_quote_df = pd.merge(df_trades_cleaned, df_quotes_cleaned, on='time', how='outer')

    lookup = df_exchanges['mic'].to_dict()
    ex_cols = ['trade_ex', 'ask_exchange', 'bid_exchange']
    trade_quote_df[ex_cols] = trade_quote_df[ex_cols].replace(lookup)

    trade_quote_df[quotes_col] = trade_quote_df[quotes_col].ffill()

    prices = pd.concat([trade_quote_df["trade_price"], trade_quote_df["bid_price"], trade_quote_df["ask_price"]])
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

    chart

def _shift_datetime(dt_str: str, period_m: int):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    
    dt_minus = dt - timedelta(minutes=period_m)
    dt_plus = dt + timedelta(minutes=period_m)
    
    return dt_minus.strftime("%Y-%m-%d %H:%M:%S"), dt_plus.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()