from pandas_datareader import data as web
import  pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

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

title = 'Portfolio Adj, Close Price History'

# Get the stocs

my_stocks = df

for c in my_stocks.columns.values:
    plt.plot(my_stocks[c],label =c)
    
plt.title(title)
plt.xlabel('Date')
plt.ylabel('Adj. Price USD ($)', fontsize=18)
plt.legend(my_stocks.columns.values, loc = 'upper left')
# plt.show()

# Show the daily simple return
print("returns_________")
returns = df.pct_change()
print(returns)

# Create and show the annualized coverience matrix

cov_matrix_annual = returns.cov()*252 # cov matrix gives directional relationship
print("cov_________")
print(cov_matrix_annual)

# Calculate the portfolio varience
port_varience = np.dot(weight.T,np.dot(cov_matrix_annual,weight))
print(port_varience)

# Calculate the portfolio volatility aka standart deviation
port_volatility = np.sqrt(port_varience)
print(port_volatility)

# Calculate the annual portfolio return
portfolioSimpleAnuualReturn = np.sum(returns.mean()*weight)*252
print(portfolioSimpleAnuualReturn)

# Show the expected annual return, volatility(risk), and variance
percent_var = str(round( port_varience , 2)*100)+ '%'
percent_volatility = str( round( port_volatility, 2 ) * 100 ) + '%'
percent_ret = str(round(portfolioSimpleAnuualReturn,2)*100)+ '%'

print('Expected annual return: ' + percent_ret)
print('Annual volatility/ risk: ' + percent_volatility )
print('Annual Varience: ' + percent_ret)

# Portfolio Optimizzation !

# Calculate the expected returns and the annualised sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for max shape ratio # sharp ratio is weight gives how much excess return you get from extra amount of volatility
# it mesures the performence of an investment compared to risk free investment

ef = EfficientFrontier(mu, S)
weight = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights) # it is going to get rid of the stock that we dont need
ef.portfolio_performance(verbose=True)

# Get the discreare allocation of each share per stock

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights,latest_prices, total_portfolio_value=15000)
allocation, leftover = da.lp_portfolio()

print('Discrete allocation:', allocation)
print('Funds remaining: ${:2f}.'.format(leftover))































