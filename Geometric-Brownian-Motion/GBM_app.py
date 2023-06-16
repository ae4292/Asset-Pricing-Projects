import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance
import statsmodels.api as sm

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
st.markdown("<h1 style='text-align: center; color: black;'>Geometric Brownian Motion Model</h1>", unsafe_allow_html=True)
st.markdown("This model allows the user to pick a ticker that can be found on Yahoo Finance for any stock they would like to make predictions on. It also allows to pick length of prediction in the future, portion of time per step, and number of simulations. This allows users to compute future variance of returns and possible expected return after a given time frame.")
st.divider()

stock = st.text_input("Ticker")
years = st.select_slider("Prediction Timeline (Years)", np.arange(0,10.01,0.25))
delta = st.text_input("Delta", "0.01")
sims = st.text_input("Simulations", "100")
option = st.selectbox("Check for Normality?", ("Yes", "No"))
st.divider()

if stock != "": 
    if option == "Yes":
        r = yfinance.Ticker(stock).history(period="1y")['Close'].pct_change().fillna(0)[1:] + 1
        fig1 = sm.qqplot(r, fit = True, line = "45")
        fig1.suptitle(f"QQ-Plot for Checking Normality of {stock} Returns")
        st.pyplot(fig1)
        st.markdown("If the returns of our given ticker follow a linear trend seen the QQ-plot, we can assume normality of return data. Thus, we can use our geometric brownian motion model to properly model future returns. Otherwise, it wouldn't be advised to use our model as it assumes normality.")  
        st.divider() 
     
    data = stockpred(stock, int(years), float(delta), int(sims))
    times = data[1]
    prices = data[0]
    fig, ax = plt.subplots()
    ax.plot(times, prices)
    ax.set_title(f"GBM for {stock}")
    ax.set_ylabel("Stock Price")
    ax.set_xlabel("Years")
    st.pyplot(fig)
    st.subheader(f"The GBM Prediction for {stock} on a {years} Year Timeline:") 
    st.markdown(f"#####        Mean: ${data[2]}")
    st.markdown(f"#####       Standard Deviation: ${data[3]}")
    st.markdown(f"#####       Confidence Interval: \${data[4][0]} to \${data[4][1]}")
    st.markdown("Given the results above, a user can make a more informed decision on how they invest their funds. With greate standard deviation comes more uncertainty of the model's prediction. Additionally, accounting for this variation in the future will allow one to make better decisions based on their risk tolerance. The mean price after a given time period is helpful in understanding the directional trend of the stock, but it shouldn't be seen as a certain prediction of future returns.")
