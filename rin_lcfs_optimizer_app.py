# RIN-LCFS Compliance Cost Optimizer - Web App Version with EIA Live Price Integration

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import streamlit as st
import requests

st.title("RIN-LCFS Compliance Cost Optimizer")
st.markdown("This tool recommends the cheapest fuel blending strategy to meet EPA RFS + LCFS compliance targets.")

# EIA series IDs for fuel prices (weekly retail data)
eia_series = {
    'Gasoline': 'PET.EMM_EPMRU_PTE_NUS_DPG.W',  # Regular retail gasoline
    'ULSD': 'PET.EMD_EPD2D_PTE_NUS_DPG.W',     # Diesel retail price
    'Ethanol': 'PET.WEPUPUS3.W',               # U.S. ethanol FOB price
    'Biodiesel': 'PET.WEBIOD3.W',              # U.S. biodiesel wholesale price
    'Renewable Diesel': 'PET.WERDUS3.W'        # Placeholder (may not exist, fallback used)
}

# EPA RIN prices (manually pulled from latest public report)
def get_live_prices():
    rin_prices = {
        'D6': 0.83,
        'D4': 1.17,
        'D5': 1.05,
        'D3': 2.48
    }
    lcfs_credit_price = 92.5
    eia_url = 'https://api.eia.gov/series/?api_key=JM9PgqPjmuvIRmsjQkkwvqk2wcBbowMAF1RLbbhU&series_id='

    prices = {}
    for name, sid in eia_series.items():
        try:
            r = requests.get(eia_url + sid)
            data = r.json()['series'][0]['data'][0][1]  # Most recent price
            prices[name] = round(data, 3)
        except:
            prices[name] = None

    return {
        'rin_prices': rin_prices,
        'lcfs_credit_price': lcfs_credit_price,
        'blend_prices': prices
    }

# Load live prices
price_data = get_live_prices()
rin_prices = price_data['rin_prices']
lcfs_credit_price = price_data['lcfs_credit_price']
blend_prices = price_data['blend_prices']

# Display live prices in sidebar
st.sidebar.header("Live Market Prices")
st.sidebar.markdown("**RIN Prices ($/RIN):**")
for k, v in rin_prices.items():
    st.sidebar.write(f"{k}: ${v:.2f}")
st.sidebar.write(f"**LCFS Credit Price:** ${lcfs_credit_price:.2f}/MT CO₂")
st.sidebar.markdown("**Blendstock Prices ($/gal):**")
for k, v in blend_prices.items():
    if v is not None:
        st.sidebar.write(f"{k}: ${v:.3f}")
    else:
        st.sidebar.write(f"{k}: Not available")

# User input
total_volume = st.number_input('Total Blend Volume (gal)', value=100000)
min_ethanol_ratio = st.slider('Minimum Ethanol Blend (%)', min_value=0.0, max_value=1.0, value=0.10)

# Blendstock pricing from EIA if available, fallback to manual
default_prices = {
    'Ethanol': 1.55,
    'Biodiesel': 4.75,
    'Renewable Diesel': 4.10,
    'Gasoline': 2.60,
    'ULSD': 3.00
}

blendstocks = pd.DataFrame({
    'name': list(default_prices.keys()),
    'base_price': [blend_prices.get(name, default_prices[name]) for name in default_prices],
    'rin_type': ['D6', 'D4', 'D4', None, None],
    'rin_yield': [1.0, 1.5, 1.7, 0.0, 0.0],
    'lcfs_credits': [0.5, 1.5, 1.6, 0.0, 0.0]
})

# Effective cost calculator
def compute_effective_cost(row):
    rin_value = 0.0
    if row['rin_type']:
        rin_price = rin_prices.get(row['rin_type'], 0)
        rin_value = row['rin_yield'] * rin_price
    lcfs_value = row['lcfs_credits'] * lcfs_credit_price
    return row['base_price'] - rin_value - lcfs_value

blendstocks['effective_price'] = blendstocks.apply(compute_effective_cost, axis=1)

# Optimization
costs = blendstocks['effective_price'].values
A_eq = [np.ones(len(costs))]
B_eq = [total_volume]
ethanol_constraint = np.array([1 if name == 'Ethanol' else 0 for name in blendstocks['name']])
A_ub = [-ethanol_constraint]
B_ub = [-total_volume * min_ethanol_ratio]
bounds = [(0, total_volume) for _ in costs]

result = linprog(c=costs, A_eq=A_eq, b_eq=B_eq, A_ub=[A_ub], b_ub=B_ub, bounds=bounds, method='highs')

if result.success:
    blendstocks['blended_volume'] = result.x
    blendstocks['blended_cost'] = blendstocks['blended_volume'] * blendstocks['effective_price']
    st.subheader("Optimized Blending Strategy")
    st.dataframe(blendstocks[['name', 'base_price', 'effective_price', 'blended_volume', 'blended_cost']])
else:
    st.error("Optimization failed. Please adjust inputs and try again.")


