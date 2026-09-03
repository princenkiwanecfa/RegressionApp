"""
Multiple Regression Tool — Streamlit app

Regress a dependent ticker's daily returns against up to 20 independent
tickers' daily returns, pulled live from Yahoo Finance via yfinance.

Run with:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

st.set_page_config(page_title="Multiple Regression Tool", layout="wide")

MAX_X = 20

st.title("📈 Multiple Regression Tool")
st.caption(
    "Regress one dependent ticker's daily returns against up to 20 independent "
    "tickers' daily returns. Data is pulled from Yahoo Finance via yfinance."
)

# ---------------------------------------------------------------------------
# Cached data fetch
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_adj_close(tickers: tuple, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download adjusted close prices for a tuple of tickers, one column per ticker."""
    data = yf.download(
        list(tickers),
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end date is exclusive
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if data.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                out[t] = data[t]["Adj Close"]
            except (KeyError, TypeError):
                continue
    else:
        # single ticker case — columns are just OHLCV, no ticker level
        t = tickers[0]
        if "Adj Close" in data.columns:
            out[t] = data["Adj Close"]

    return pd.DataFrame(out)


def to_returns_pct(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns in percentage terms."""
    return prices.pct_change().mul(100).dropna(how="all")


def cumulative_return_pct(returns_pct: pd.Series) -> pd.Series:
    return ((1 + returns_pct / 100).cumprod() - 1) * 100


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
today = dt.date.today()
default_start = today - dt.timedelta(days=5 * 365)

with st.form("inputs_form"):
    st.subheader("Date Range")
    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input("Start date", value=default_start, max_value=today)
    with d2:
        end_date = st.date_input("End date", value=today, max_value=today)

    st.subheader("Dependent Variable (Y)")
    y_ticker = st.text_input("Yahoo Finance ticker", value="^GSPC").strip().upper()

    st.subheader(f"Independent Variables (X) — up to {MAX_X}")
    st.caption("Add or remove rows. Blank rows are ignored.")
    default_x = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "^TNX"]})
    x_editor = st.data_editor(
        default_x,
        num_rows="dynamic",
        width="stretch",
        key="x_ticker_editor",
    )

    include_const = st.checkbox("Include intercept (constant)", value=True)
    submitted = st.form_submit_button("Run Regression", type="primary")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if submitted:
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    x_tickers = [str(t).strip().upper() for t in x_editor["Ticker"].tolist() if str(t).strip()]
    x_tickers = list(dict.fromkeys(x_tickers))  # de-dupe, preserve order

    if not y_ticker:
        st.error("Please enter a dependent variable ticker.")
        st.stop()
    if not x_tickers:
        st.error("Please enter at least one independent variable ticker.")
        st.stop()
    if len(x_tickers) > MAX_X:
        st.error(f"Please limit independent variables to {MAX_X}. You entered {len(x_tickers)}.")
        st.stop()
    if y_ticker in x_tickers:
        st.error("The dependent variable ticker cannot also appear as an independent variable.")
        st.stop()

    all_tickers = tuple([y_ticker] + x_tickers)

    with st.spinner(f"Downloading data for {len(all_tickers)} ticker(s)..."):
        try:
            prices = fetch_adj_close(all_tickers, start_date, end_date)
        except Exception as e:
            st.error(f"Failed to download data: {e}")
            st.stop()

    missing = [t for t in all_tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        st.error(f"No data returned for: {', '.join(missing)}. Check the ticker symbol(s) and try again.")
        st.stop()

    returns = to_returns_pct(prices[list(all_tickers)]).dropna()

    if returns.shape[0] < len(x_tickers) + 2:
        st.error(
            f"Not enough overlapping trading days ({returns.shape[0]}) for {len(x_tickers)} "
            "independent variables. Widen the date range or reduce the number of X tickers."
        )
        st.stop()

    y = returns[y_ticker]
    X = returns[x_tickers]

    X_model = sm.add_constant(X) if include_const else X
    model = sm.OLS(y, X_model, missing="drop").fit()

    st.success(
        f"Regression complete — {returns.shape[0]} overlapping trading days, "
        f"{start_date} to {end_date}."
    )

    # -----------------------------------------------------------------
    # Overall regression stats
    # -----------------------------------------------------------------
    st.markdown("## Regression Output")

    dw_stat = durbin_watson(model.resid)

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    stats_col1.metric("R-squared", f"{model.rsquared:.4f}")
    stats_col2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
    stats_col3.metric("F-statistic", f"{model.fvalue:.3f}")
    stats_col4.metric("Prob (F-stat)", f"{model.f_pvalue:.4g}")

    stats_col5, stats_col6, stats_col7, stats_col8 = st.columns(4)
    stats_col5.metric("No. Observations", f"{int(model.nobs)}")
    stats_col6.metric("Durbin-Watson", f"{dw_stat:.3f}")
    stats_col7.metric("AIC", f"{model.aic:.2f}")
    stats_col8.metric("BIC", f"{model.bic:.2f}")

    st.markdown("### Coefficients")
    coef_table = pd.DataFrame(
        {
            "Coefficient": model.params,
            "Std. Error": model.bse,
            "t-stat": model.tvalues,
            "P>|t|": model.pvalues,
            "CI Lower (95%)": model.conf_int()[0],
            "CI Upper (95%)": model.conf_int()[1],
        }
    )
    coef_table.index = coef_table.index.map(lambda n: "Intercept" if n == "const" else n)

    def sig_stars(p):
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.10:
            return "*"
        return ""

    coef_table["Sig."] = coef_table["P>|t|"].apply(sig_stars)
    st.dataframe(
        coef_table.style.format(
            {
                "Coefficient": "{:.4f}",
                "Std. Error": "{:.4f}",
                "t-stat": "{:.3f}",
                "P>|t|": "{:.4g}",
                "CI Lower (95%)": "{:.4f}",
                "CI Upper (95%)": "{:.4f}",
            }
        ),
        width="stretch",
    )
    st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.10")

    if len(x_tickers) > 1:
        st.markdown("### Multicollinearity (Variance Inflation Factor)")
        try:
            vif_data = pd.DataFrame(
                {
                    "Variable": X.columns,
                    "VIF": [
                        variance_inflation_factor(X.assign(const=1.0).values, i)
                        for i in range(len(X.columns))
                    ],
                }
            ).set_index("Variable")
            st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), width="stretch")
            st.caption("VIF > 10 typically signals problematic multicollinearity between X variables.")
        except Exception:
            st.info("VIF could not be computed (likely due to perfectly collinear variables).")

    with st.expander("Full statsmodels summary (text)"):
        st.text(str(model.summary()))

    # -----------------------------------------------------------------
    # Cumulative returns chart
    # -----------------------------------------------------------------
    st.markdown("## Cumulative Returns")
    st.caption(f"All series rebased to 0% at {returns.index[0].date()}.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=cumulative_return_pct(y),
            name=f"{y_ticker} (Y)",
            line=dict(width=3, dash="solid"),
        )
    )
    for t in x_tickers:
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=cumulative_return_pct(returns[t]),
                name=t,
                line=dict(width=1.5),
            )
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=550,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("View underlying daily returns (%)"):
        st.dataframe(returns.style.format("{:.3f}"), width="stretch")

else:
    st.info("Set your date range, Y ticker, and X tickers above, then click **Run Regression**.")
