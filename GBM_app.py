import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yfinance
import time
import matplotlib.animation as animation
import streamlit.components.v1 as components


def stockpred (stock,years = 1, dt = 0.01, sims = 100):
    stock = yfinance.Ticker(stock).history(period="1y")['Close']
    rstock = stock.pct_change().fillna(0)[1:] + 1
    ln_mean = np.mean(np.log(rstock))
    ln_sd = np.std(np.log(rstock))
    S0 = stock[-1]
    steps = int(years/dt)
    sigma = ln_sd * np.sqrt(1/dt)
    mu = ln_mean / dt + sigma ** 2 / 2


    St = np.exp((mu -sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size = (sims,steps)).T)
    St = np.vstack([np.ones(sims), St])
    St = S0 * St.cumprod(axis=0)

    last_date = St.shape[0]
    last_prices = St[last_date - 1]
    mean = np.round(np.mean(last_prices),2)
    sd = np.round(np.std(last_prices),2)
    CI = [np.round(np.percentile(last_prices, 2.5),2),np.round(np.percentile(last_prices, 97.5),2)]
    time = np.linspace(0,years,steps+1)
    tt = np.full(shape=(sims,steps+1), fill_value=time).T
    
    return St, tt, mean, sd, CI
st.title("Adrian's Cool GBM Model")
st.markdown("This model runs 100 simulations with a delta of 0.01")
stock = st.text_input("Ticker")
years = st.text_input("Prediction Timeline (Years)")

if stock != "":
    if years != "":
        data = stockpred(stock, int(years))
    else:
        data = stockpred(stock)
    times = data[1]
    prices = data[0]
    fig, ax = plt.subplots()
    ax.plot(times, prices)
    ax.set_title(f"GBM for {stock}")
    ax.set_ylabel("Stock Price")
    ax.set_xlabel("Years")
    st.pyplot(fig)
    n = "\n"
    st.subheader(f"The GBM Prediction for {stock} on a {years} Year Timeline had:") 
    st.markdown(f"######        Mean: ${data[2]}")
    st.markdown(f"######        Standard Deviation: ${data[3]}")
    st.markdown(f"######        Confidence Interval: \${data[4][0]} to \${data[4][1]}")