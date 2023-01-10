# About

The code attempts to quantify the "edge" associated with a short volatility strategy expressed using a short at-the-money straddle.

For a range of future volatility forecasts, the code returns a distribution of returns.

Monte Carlo method is used to simulate the underlying prices with forecasted volatility values as input. For each of the simulated underlying price, Black's model is used to calculate the future price of the straddle.

The output is a P/L table for the volatility forecasts. 

Important Note: The code uses 5Paisa API for fetching real-time prices. It can be replaced with any other API.
