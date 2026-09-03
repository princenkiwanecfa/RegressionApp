# Multiple Regression Tool

A Streamlit app that regresses one dependent ticker's daily returns against
up to 20 independent tickers' daily returns, using live data from Yahoo
Finance (`yfinance`).

## Features

- **Date range picker** — calendar dropdowns for start/end date. Defaults to
  the last 5 years (start) through today (end).
- **Y variable** — enter any valid Yahoo Finance ticker; its adjusted close
  is converted to daily % returns over the selected range.
- **X variables** — an editable table where you can add/remove up to 20
  tickers; each is converted to daily % returns the same way.
- **Regression output** — R², Adjusted R², F-statistic & its p-value,
  N observations, Durbin-Watson, AIC, BIC, and a full coefficient table
  (coefficient, std. error, t-stat, p-value, 95% CI, significance stars).
  A Variance Inflation Factor (VIF) table is shown when there's more than
  one X variable, to flag multicollinearity. The full statsmodels text
  summary is available in an expander.
- **Cumulative returns chart** — an interactive Plotly line chart of the
  dependent and every independent variable's cumulative return over the
  selected date range.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Notes

- Returns are simple daily percentage changes in adjusted close price
  (dividend/split-adjusted), not log returns.
- Only trading days common to *all* selected tickers are used (an inner
  join), so the regression sample may be shorter than the full calendar
  range if tickers trade on different exchanges/holidays.
- The intercept (constant) can be toggled on/off.
