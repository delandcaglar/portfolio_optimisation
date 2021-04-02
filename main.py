from pandas_datareader import data as web
import  pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


plt.style.use('fivethirtyeight')
print("done")

# Get the stock symbols / tickers in the portfolio
# FAANG

assets = ['FB','AMZN','AAPL','NFLX','GOOG']

# Assign wights to the stocks.

weight = np.array([0.2,0.2,0.2,0.2,0.2])

# Get the stock/portfolio starting date
stockStartDate = '2013-01-01'

# Get the stocks ending date(today)
today = datetime.today().strftime('%Y-%m-%d')
print(today)

# Create a dateframe to store the adjusted close proce of the stocks
df = pd.DataFrame()

# Store the adjusted close price of the stock into the df

for stock in assets:
    df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate, end=today)['Adj Close']

# Show the dataframe

print(df)

# Visually show the stock/ portfolio





















