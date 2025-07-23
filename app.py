# RIN-LCFS Compliance Cost Optimizer – Streamlit App (Auto-Updating Free Data)

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import streamlit as st
import requests
from datetime import datetime

st.title("RIN-LCFS Compliance Cost Optimizer")
st.markdown("This tool recommends the cheapest fuel blending strategy to meet EPA RFS + LCFS compliance targets.")

# ------------------------------
# 🔑 CONFIG
# ------------------------------
API_KEY = "JM9PgqPjmuvIRmsjQkkwvqk2wcBbowMAF1RLbbhU"  # put in st.secrets["EIA_KEY"] for prod
REFRESH_TTL_SEC = 1800  # 30 minutes cache for free data

# EIA weekly series IDs ($/gal). Some don't exist -> use None and fallback price
EIA_SERIES = {
    "Gasoline": "PET.EMM_EPMRU_PTE_NUS_DPG.W",   # Regular retail gasoline
    "ULSD": "PET.EMD_EPD2D_PTE_NUS_DPG.W",       # Diesel retail price
    "Ethanol": "PET.WEPUPUS3.W",                 # U.S. ethanol FOB price
    "Biodiesel": "PET.WEBIOD3.W",                # Biodiesel wholesale price (if 404 -> fallback)
    "Renewable Diesel": None                       # No public EIA series => fallback
}

# Default manual prices if API fails or series missing ($/gal)
DEFAULT_PRICES = {
    "Ethanol": 1.55,
    "Biodiesel": 4.75,
    "Renewable Diesel": 4.10,
    "Gasoline": 2.60,
    "ULSD": 3.00
}

# Static example RIN prices & LCFS (free sources are lagged; we keep manual inputs)
DEFAULT_RIN_PRICES = {"D6": 0.83, "D4": 1.17, "D5": 1.05, "D3": 2.48}
DEFAULT_LCFS_PRICE = 92.5  # $/MT CO2

# ------------------------------
# 🛰️ DATA FETCH HELPERS
# ------------------------------
@st.cache_data(ttl=REFRESH_TTL_SEC)
def fetch_eia_price(series_id: str, api_key: str):
    if series_id is None:
        return None
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    js = r.json()
    return float(js["series"][0]["data"][0][1])

@st.cache_data(ttl=REFRESH_TTL_SEC)
def get_live_prices():
    prices = {}
    failures = {}
    for name, sid in EIA_SERIES.items():
        try:
            prices[name] = fetch_eia_price(sid, API_KEY)
        except Exception as e:
            prices[name] = None
            failures[name] = str(e)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return prices, failures, ts

# ------------------------------
# 🔁 REFRESH CONTROL
# ------------------------------
colr1, colr2 = st.sidebar.columns([1,1])
with colr1:
    if st.button("↻ Refresh prices"):
        fetch_eia_price.clear()
        get_live_prices.clear()
        st.experimental_rerun()
with colr2:
    st.write("")

# ------------------------------
# 📡 GET DATA
# ------------------------------
blend_prices, failures, fetched_at = get_live_prices()
rin_prices = DEFAULT_RIN_PRICES.copy()
lcfs_credit_price = DEFAULT_LCFS_PRICE

# ------------------------------
# 🧰 SIDEBAR INPUTS / INFO
# ------------------------------
st.sidebar.header("Live / Fallback Prices")
st.sidebar.caption(f"Fetched: {fetched_at}  (cache {REFRESH_TTL_SEC//60} min)")

st.sidebar.subheader("RIN Prices ($/RIN)")
for k, v in rin_prices.items():
    rin_prices[k] = st.sidebar.number_input(f"{k} RIN", value=float(v))

lcfs_credit_price = st.sidebar.number_input("LCFS Credit ($/MT CO₂)", value=float(lcfs_credit_price))

st.sidebar.subheader("Blendstock Prices ($/gal)")
final_prices = {}
for name, default_val in DEFAULT_PRICES.items():
    live_val = blend_prices.get(name)
    if live_val is None:
        label = f"{name} (fallback)"
        final_prices[name] = st.sidebar.number_input(label, value=float(default_val))
    else:
        label = f"{name} (live)"
        final_prices[name] = st.sidebar.number_input(label, value=float(live_val))

# Show any failures
if failures:
    with st.sidebar.expander("API Fetch Errors", expanded=False):
        for n, msg in failures.items():
            st.write(f"**{n}**: {msg}")

# ------------------------------
# 🧮 USER BLEND CONSTRAINTS
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    total_volume = st.number_input("Total Blend Volume (gal)", value=100_000)
with col2:
    min_ethanol_ratio = st.slider("Minimum Ethanol Blend (%)", min_value=0.0, max_value=1.0, value=0.10)

# ------------------------------
# 📑 BUILD BLENDSTOCK TABLE
# ------------------------------
blendstocks = pd.DataFrame({
    'name': list(DEFAULT_PRICES.keys()),
    'base_price': [final_prices[n] for n in DEFAULT_PRICES.keys()],
    'rin_type': ['D6', 'D4', 'D4', None, None],
    'rin_yield': [1.0, 1.5, 1.7, 0.0, 0.0],
    'lcfs_credits': [0.5, 1.5, 1.6, 0.0, 0.0]
})

# ------------------------------
# 💵 EFFECTIVE COST CALC
# ------------------------------
def compute_effective_cost(row):
    rin_value = row['rin_yield'] * rin_prices.get(row['rin_type'], 0) if row['rin_type'] else 0.0
    lcfs_value = row['lcfs_credits'] * lcfs_credit_price
    return row['base_price'] - rin_value - lcfs_value

blendstocks['effective_price'] = blendstocks.apply(compute_effective_cost, axis=1)

# ------------------------------
# 📉 OPTIMIZATION
# ------------------------------
costs = blendstocks['effective_price'].values
A_eq = [np.ones(len(costs))]
B_eq = [total_volume]

ethanol_mask = np.array([1 if n == 'Ethanol' else 0 for n in blendstocks['name']])
A_ub = [-ethanol_mask]
B_ub = [-total_volume * min_ethanol_ratio]

bounds = [(0, total_volume) for _ in costs]

result = linprog(c=costs, A_eq=A_eq, b_eq=B_eq, A_ub=[A_ub], b_ub=B_ub, bounds=bounds, method='highs')

# ------------------------------
# 📊 RESULTS
# ------------------------------
if result.success:
    blendstocks['blended_volume'] = result.x
    blendstocks['blended_cost'] = blendstocks['blended_volume'] * blendstocks['effective_price']
    st.subheader("Optimized Blending Strategy")
    st.dataframe(blendstocks[['name', 'base_price', 'effective_price', 'blended_volume', 'blended_cost']])
    st.write(f"**Total Cost:** ${blendstocks['blended_cost'].sum():,.2f}")
else:
    st.error("Optimization failed. Adjust inputs and try again.")

st.caption("If a price couldn't be fetched from EIA, a fallback manual input is used. Click 'Refresh prices' to pull fresh data.")
