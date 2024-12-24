import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import statistics as sd
import streamlit as st
import altair as alt


def random_walk(dt, mu, K, sigma, nsteps, nruns):
    z = np.random.randn
    yT = x0 * np.ones((nsteps +1, nruns))
    for step in range(1, nsteps + 1):
        # Update stock price with constraint to stay non-negative
        yT[step, :] = yT[step - 1, :]*(1+ (mu*(dt)) + sigma*np.sqrt(dt)*z(nruns))
        yT[step, :] = np.maximum(yT[step, :], 0)  # Ensure non-negative prices

    # Calculate call and put payoffs
    c_payoff = np.maximum(yT[nsteps] - K, 0)
    p_payoff = np.maximum(K - yT[nsteps], 0)

    return yT, c_payoff, p_payoff



##x0 = 100
##sigma = 0.2
##mu = 0.05
##nsteps = 32
##T = 10
##dt = T/nsteps
##nruns = 100
##K = 100
##r=mu



st.set_page_config(
    page_title="Euler-Maruyama Monte-Carlo Simulations",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊📈 Euler-Maruyama Monte-Carlo Simulations")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/danieldmalone/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Daniel Malone`</a>', unsafe_allow_html=True)

    x0 = st.number_input("Asset Start Price", value=100.0)
    K = st.number_input("Strike Price", value=100.0)
    T = st.number_input("Time to Maturity (Years)", value=1.0)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    mu = st.number_input("Risk-Free Interest Rate", value=0.05)
    nruns = st.number_input("Number of runs", value=2)
    nsteps = st.number_input("Number of Steps", value=4)


dt = T/nsteps
r= mu
yT, c_payoff, p_payoff = random_walk(dt, mu, K, sigma, nsteps, nruns)

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ff3c00; /* Light red background */
    color: White; /* White font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)




# Table of Inputs
st.title("Input Data")

input_data = {
    "Current Asset Price": [x0],
    "Strike Price": [K],
    "Time to Maturity (Years)": [T],
    "Volatility (σ)": [sigma],
    "Risk-Free Interest Rate": [mu],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)


#Simulations graph
st.title("Euler-Maruyama Monte Carlo Simulations")
st.line_chart(yT)


#Monte Carlo options
EV2 = x0 * np.exp(mu*T)
mean = np.mean(yT[nsteps])
Call_0 = np.exp(-r * T) * np.mean(c_payoff)
Put_0 = np.exp(-r * T) * np.mean(p_payoff)

#black scholes merton model
d1 = (np.log(x0/K) + (r + (sigma**2/2))*T)/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_price = x0* norm.cdf(d1) - (K*np.exp(-r*T) * norm.cdf(d2))
put_price = (K*np.exp(-r*T) * norm.cdf(-d2)) - x0*norm.cdf(-d1)


# Display Call & Put Values given by simulations in colored tables
st.title("Call & Put Prices given by Monte Carlo Simulations")
st.info("The prices given by the Simulations (taking the mean payoff value and using 𝐸[𝑓(𝑆𝑇)]𝑒−𝑟(𝑇−𝑡)) will converge towards the Black-Scholes prices as the number of simulations increases")

col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${Call_0:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${Put_0:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display Call and Put Values in colored tables given by BS model

st.title("Black Scholes Call & Put Prices")


col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

