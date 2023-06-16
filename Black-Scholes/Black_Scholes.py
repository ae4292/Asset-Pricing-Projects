import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import yfinance
from datetime import datetime
from datetime import date
import streamlit as st
import re

def price(S,K,r,t,sigma, call = True):
    d1 = (np.log(S/K) + (r +.5*sigma**2)*t)/(sigma * np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)

    Call_price = S * norm.cdf(d1,0,1) - K * np.exp(-r*t)*norm.cdf(d2,0,1)
    Put_price = K * np.exp(-r*t)* norm.cdf(-d2,0,1) - S*norm.cdf(-d1,0,1)

    if call == True:
        return Call_price
    else:
        return Put_price

r = yfinance.Ticker("^TNX").history(period="1y")['Close'][-1] / 100
dates = yfinance.Ticker("SPY").options

def ret_type(symbol,ticker):
    string1 = symbol.strip(ticker)
    string1 = re.sub(r'[0-9]', '', string1)
    return string1

#Price array has an error somewhere as it gives put values for call = True and vice versa
def price_arrays(stock, strike, exp_date, symb, res_chains):
    if symb != "P":
        call = False
    else:
        call = True
    
    stock_price = yfinance.Ticker(stock).history(period="1y")['Close'][-1]

    real = res_chains[res_chains["symbols"] != symb]["lastPrice"]
    prices = res_chains[res_chains["symbols"] != symb]["strike"]
    imp_vol = np.max(res_chains[(res_chains["strike"] == float(strike)) & (res_chains["symbols"] != symb)]["impliedVolatility"])

    date_today = datetime(date.today().year, date.today().month, date.today().day)
    fut_datetime = datetime.strptime(exp_date,'%Y-%m-%d')
    diff = fut_datetime - date_today
    t = diff.days/365
    
    symb_price_array = np.array([price(stock_price,pr,r,t,imp_vol, call) for pr in prices])
    return prices, real, prices, symb_price_array

st.markdown("<h1 style='text-align: center; color: black;'>Black Scholes Option Pricing Model</h1>", unsafe_allow_html=True)
st.markdown("This model allows the user to pick a ticker that can be found on Yahoo Finance for any stock they would like to make predictions on its options.")
st.divider()
stock = st.text_input("Ticker")

r = yfinance.Ticker("^TNX").history(period="1y")['Close'][-1] / 100
if stock != "":
    dates = yfinance.Ticker(stock).options

    exp_date = st.selectbox("Expiration Date", dates)
    option_type = st.selectbox("Stock Option Type", ["","C", "P"])

    chain = yfinance.Ticker(stock).option_chain(exp_date)
    chains = list()
    for i in np.arange(len(chain)):
        chains.append(chain[i])
    res_chains = pd.concat(chains)

    symbols = res_chains["contractSymbol"]
    res_chains["symbols"] = [ret_type(x,stock) for x in symbols]

    if option_type != "":

        front = np.array([""])
        strikes = res_chains[res_chains["symbols"] == option_type]["strike"]
        strike = st.selectbox("Strike", np.append(front,strikes))
        if strike != "":
            data = price_arrays(stock, strike, exp_date, option_type, res_chains)

            real_strikes = data[0]
            real_opts = data[1]
            sim_strikes = data[2]
            sim_opts = data[3]

            fig, ax = plt.subplots()
            
            ax.plot(real_strikes, real_opts, label = "Market Prices")
            ax.plot(sim_strikes,sim_opts, label =  "Black Scholes")
            ax.axvline(float(strike), color = "red", label = "Strike")
            ax.legend()
            ax.set_xlabel("Strike Price ($)")
            ax.set_ylabel("Option Value ($)")
            st.divider()
            st.subheader("Change in Option Value given a different Strike Price")
            st.pyplot(fig)

            index = np.where(real_strikes == float(strike))
            index1 = np.where(sim_strikes == float(strike))
            st.divider()
            diff = abs(np.round(sim_opts[index][0],2) - real_opts.iloc[index1].values[0])
            st.subheader("Price Comparison of Model and Market Value")
            st.markdown(f"Black Scholes Price = ${np.round(sim_opts[index][0],2)}")
            st.markdown(f"Market Value = ${real_opts.iloc[index1].values[0]}")
            st.markdown(f"Difference = {diff}")
            st.divider()
            st.subheader("Analysis")
            st.markdown("While the Blak Scholes isn't perfect for every type fo option, it is definitely close to market prices. I put the market prices and option to see if they align at the strike price. In some cases the Black Scholes Model follows markets trends exactly. This is most true for options that are very close to expiration since there is no room for price speculation. As seen above, Black Scholes is still very appilicable when pricing options.")

    
    
