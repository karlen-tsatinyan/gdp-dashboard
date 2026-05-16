#to see in browser

#streamlit run /workspaces/gdp-dashboard/x.py --server.enableCORS=false --server.enableXsrfProtection=false


import yfinance as yf
df = yf.download("AAPL", start="2025-01-01")
print(df.head())
