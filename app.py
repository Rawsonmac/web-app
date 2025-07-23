import pandas as pd
import numpy as np
from scipy.optimize import linprog
import streamlit as st
import requests
from requests.exceptions import RequestException
import os
import json

st.title("RIN-LCFS Compliance Cost Optimizer")
st.markdown("This tool recommends the cheapest fuel blending strategy to meet EPA RFS + LCFS compliance targets.")

# EIA series IDs
eia_series = {
    'Gasoline': 'PET.EMM_EPMRU_PTE_NUS_DPG.W',
    'ULSD': 'PET.EMD_EPD2D_PTE_NUS_DPG.W',
    'Ethanol': 'PET.WEPUPUS3.W',
    'Biodiesel': 'PET.WEBIOD3.W',
    'Renewable Diesel': 'PET.WERDUS3.W'
}

# Default prices
default_prices = {
    'Ethanol': 1.55,
    'Biodiesel': 4.75,
    'Renewable Diesel': 4.10,
    'Gasoline': 2.60,
    'ULSD': 3.00
}

# Fetch live prices
@st.cache_data(ttl=3600)
def get_live_prices():
    rin_prices = {
        'D6': 0.83,
        'D4': 1.17,
        'D5': 1.05,
        'D3': 2.48
    }
    lcfs_credit_price = 92.5
    eia_url = f"https://api.eia.gov/series/?api_key={os.getenv('EIA_API_KEY', 'YOUR_EIA_API_KEY')}&series_id="
    prices = {}
    
    for name, sid in eia_series.items():
        try:
            r = requests.get(eia_url + sid, timeout=5)
            r.raise_for_status()
            data = r.json().get('series', [{}])[0].get('data', [[]])[0]
            if len(data) > 1:
                prices[name] = round(data[1], 3)
                st.sidebar.success(f"Fetched {name} price: ${prices[name]:.3f}")
            else:
                raise ValueError("No data available")
        except (RequestException, ValueError, IndexError) as e:
            st.sidebar.warning(f"Failed to fetch {name} price: {e}. Using default.")
            prices[name] = default_prices.get(name)
    
    return {
        'rin_prices': rin_prices,
        'lcfs_credit_price': lcfs_credit_price,
        'blend_prices': prices
    }

# Load prices
price_data = get_live_prices()
rin_prices = price_data['rin_prices']
lcfs_credit_price = price_data['lcfs_credit_price']
blend_prices = price_data['blend_prices']

# Sidebar: Price overrides
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

st.sidebar.subheader("Override RIN/LCFS Prices")
rin_d6 = st.sidebar.number_input('D6 RIN Price ($/RIN)', value=rin_prices['D6'], step=0.01)
rin_d4 = st.sidebar.number_input('D4 RIN Price ($/RIN)', value=rin_prices['D4'], step=0.01)
rin_d5 = st.sidebar.number_input('D5 RIN Price ($/RIN)', value=rin_prices['D5'], step=0.01)
rin_d3 = st.sidebar.number_input('D3 RIN Price ($/RIN)', value=rin_prices['D3'], step=0.01)
lcfs_credit_price = st.sidebar.number_input('LCFS Credit Price ($/MT CO₂)', value=lcfs_credit_price, step=0.1)
rin_prices = {'D6': rin_d6, 'D4': rin_d4, 'D5': rin_d5, 'D3': rin_d3}

import json

with open('blendstocks.json', 'r') as f:
    blendstock_data = json.load(f)

    
blendstocks = pd.DataFrame({
    'name': list(blendstock_data.keys()),
    'base_price': [blend_prices.get(name, data['base_price']) for name, data in blendstock_data.items()],
    'rin_type': [data['rin_type'] for data in blendstock_data.values()],
    'rin_yield': [data['rin_yield'] for data in blendstock_data.values()],
    'lcfs_credits': [data['lcfs_credits'] for data in blendstock_data.values()]
})

# Blendstock selection
st.subheader("Select Blendstocks")
selected_blendstocks = st.multiselect("Include in Optimization", blendstocks['name'].tolist(), default=blendstocks['name'].tolist())
blendstocks = blendstocks[blendstocks['name'].isin(selected_blendstocks)]

if blendstocks.empty:
    st.error("Please select at least one blendstock.")
    st.stop()

# Effective cost calculator
def compute_effective_cost(row):
    rin_value = 0.0
    if row['rin_type']:
        rin_price = rin_prices.get(row['rin_type'], 0)
        rin_value = row['rin_yield'] * rin_price
    lcfs_value = row['lcfs_credits'] * lcfs_credit_price
    return row['base_price'] - rin_value - lcfs_value

blendstocks['effective_price'] = blendstocks.apply(compute_effective_cost, axis=1)

# User input
st.subheader("Input Parameters")
st.markdown("""
- **Total Blend Volume**: Enter the total fuel volume (in gallons) to blend.
- **Minimum Ethanol Blend**: Specify the minimum percentage of ethanol required (e.g., 10% for E10).
""")
total_volume = st.number_input('Total Blend Volume (gal)', min_value=1.0, max_value=1000000.0, value=100000.0, step=1000.0)
min_ethanol_ratio = st.slider('Minimum Ethanol Blend (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.1) / 100.0

if total_volume <= 0:
    st.error("Total volume must be positive.")
    st.stop()
if min_ethanol_ratio > 1 or min_ethanol_ratio < 0:
    st.error("Ethanol blend ratio must be between 0% and 100%.")
    st.stop()

# Optimization
ethanol_constraint = np.array([1 if name == 'Ethanol' else 0 for name in blendstocks['name']])
costs = blendstocks['effective_price'].values
A_eq = [np.ones(len(costs))]
b_eq = [total_volume]
A_ub = [
    -ethanol_constraint,
    [1 if name == 'Ethanol' else 0 for name in blendstocks['name']],
    [1 if name == 'Biodiesel' else 0 for name in blendstocks['name']]
]
b_ub = [
    -total_volume * min_ethanol_ratio,
    total_volume * 0.15,
    total_volume * 0.20
]
bounds = [(0, total_volume) for _ in costs]

# RFS constraint (example: at least 10% D6 RINs)
if 'Ethanol' in blendstocks['name'].values:
    d6_rin_yield = blendstocks[blendstocks['rin_type'] == 'D6']['rin_yield'].values[0]
    A_ub.append([d6_rin_yield if name == 'Ethanol' else 0 for name in blendstocks['name']])
    b_ub.append(total_volume * 0.10)

result = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Output
if result.success:
    blendstocks['blended_volume'] = result.x
    blendstocks['blended_cost'] = blendstocks['blended_volume'] * blendstocks['effective_price']
    
    # Format DataFrame
    display_df = blendstocks[['name', 'base_price', 'effective_price', 'blended_volume', 'blended_cost']].copy()
    display_df['base_price'] = display_df['base_price'].round(3)
    display_df['effective_price'] = display_df['effective_price'].round(3)
    display_df['blended_volume'] = display_df['blended_volume'].round(2)
    display_df['blended_cost'] = display_df['blended_cost'].round(2)
    
    st.subheader("Optimized Blending Strategy")
    st.dataframe(display_df.style.format({
        'base_price': "${:.3f}",
        'effective_price': "${:.3f}",
        'blended_volume': "{:,.2f} gal",
        'blended_cost': "${:,.2f}"
    }))
    
    # Summary
    total_cost = blendstocks['blended_cost'].sum()
    total_rins = (blendstocks['blended_volume'] * blendstocks['rin_yield']).sum()
    total_lcfs = (blendstocks['blended_volume'] * blendstocks['lcfs_credits']).sum()
    st.subheader("Summary")
    st.write(f"**Total Cost:** ${total_cost:,.2f}")
    st.write(f"**Total RINs Generated:** {total_rins:,.2f}")
    st.write(f"**Total LCFS Credits:** {total_lcfs:,.2f} MT CO₂")
    
    # Pie chart
    non_zero_volumes = blendstocks[blendstocks['blended_volume'] > 0]
    if not non_zero_volumes.empty:
        st.subheader("Blend Composition")
        chart_data = {
            "type": "pie",
            "data": {
                "labels": non_zero_volumes['name'].tolist(),
                "datasets": [{
                    "data": non_zero_volumes['blended_volume'].tolist(),
                    "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"],
                }]
            },
            "options": {
                "plugins": {
                    "legend": {"position": "top"},
                    "title": {"display": True, "text": "Blend Composition by Volume"}
                }
            }
        }
        st.write("The following chart shows the proportion of each blendstock in the optimized blend:")
        st.json(chart_data)  # Note: Streamlit may need a charting library for this
else:
    st.error("Optimization failed. Please adjust inputs and try again.")

# Reset button
if st.button("Reset Inputs"):
    st.experimental_rerun()
