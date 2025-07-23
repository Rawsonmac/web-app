# RIN-LCFS Compliance Cost Optimizer – Streamlit App (Auto-Updating Free Data + UX Improvements)

import pandas as pd
import numpy as np
from scipy.optimize import linprog
import streamlit as st
import requests
from datetime import datetime
import altair as alt

st.set_page_config(page_title="RIN / LCFS Optimizer", layout="wide")

st.title("RIN-LCFS Compliance Cost Optimizer")
st.markdown(
    """
    Find the **cheapest blend** (Ethanol, Biodiesel, Renewable Diesel, Gasoline, ULSD) that meets your compliance constraints (RINs / LCFS / ethanol %).
    Prices auto-pull from **EIA** where available; everything else can be overridden.
    """
)

# ------------------------------
# 🔑 CONFIG
# ------------------------------
API_KEY = "JM9PgqPjmuvIRmsjQkkwvqk2wcBbowMAF1RLbbhU"  # move to st.secrets["EIA_KEY"] in prod
REFRESH_TTL_SEC = 1800  # 30 min cache

EIA_SERIES = {
    "Gasoline": "PET.EMM_EPMRU_PTE_NUS_DPG.W",   # Regular retail gasoline
    "ULSD": "PET.EMD_EPD2D_PTE_NUS_DPG.W",       # Diesel retail price
    "Ethanol": "PET.WEPUPUS3.W",                 # Ethanol FOB
    "Biodiesel": "PET.WEBIOD3.W",                # Biodiesel wholesale (if 404 -> fallback)
    "Renewable Diesel": None                      # No public series -> fallback
}

DEFAULT_PRICES = {
    "Ethanol": 1.55,
    "Biodiesel": 4.75,
    "Renewable Diesel": 4.10,
    "Gasoline": 2.60,
    "ULSD": 3.00
}

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
st.sidebar.header("Data & Inputs")
colr1, colr2 = st.sidebar.columns([1,1])
with colr1:
    if st.button("↻ Refresh prices"):
        fetch_eia_price.clear(); get_live_prices.clear(); st.experimental_rerun()
with colr2:
    st.write("")

# ------------------------------
# 📡 GET DATA
# ------------------------------
blend_prices, failures, fetched_at = get_live_prices()
rin_prices = DEFAULT_RIN_PRICES.copy()
lcfs_credit_price = DEFAULT_LCFS_PRICE

st.sidebar.caption(f"Fetched: {fetched_at}  (cache {REFRESH_TTL_SEC//60} min)")

# ------------------------------
# 🧰 PRICE INPUTS
# ------------------------------
st.sidebar.subheader("RIN Prices ($/RIN)")
for k, v in rin_prices.items():
    rin_prices[k] = st.sidebar.number_input(f"{k} RIN", value=float(v))

lcfs_credit_price = st.sidebar.number_input("LCFS Credit ($/MT CO₂)", value=float(lcfs_credit_price))

st.sidebar.subheader("Blendstock Prices ($/gal)")
final_prices = {}
for name, default_val in DEFAULT_PRICES.items():
    live_val = blend_prices.get(name)
    tag = "live" if live_val is not None else "fallback"
    final_prices[name] = st.sidebar.number_input(f"{name} ({tag})", value=float(live_val or default_val))

if failures:
    with st.sidebar.expander("API Fetch Errors", expanded=False):
        for n, msg in failures.items():
            st.write(f"**{n}**: {msg}")

# ------------------------------
# 🧮 USER BLEND CONSTRAINTS
# ------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    total_volume = st.number_input("Total Blend Volume (gal)", value=100_000, min_value=0)
with col2:
    min_ethanol_ratio = st.slider("Min Ethanol %", min_value=0.0, max_value=1.0, value=0.10)
with col3:
    show_charts = st.checkbox("Show charts", value=True)

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

# Effective cost
def compute_effective_cost(row):
    rin_value = row['rin_yield'] * rin_prices.get(row['rin_type'], 0) if row['rin_type'] else 0.0
    lcfs_value = row['lcfs_credits'] * lcfs_credit_price
    return row['base_price'] - rin_value - lcfs_value

blendstocks['effective_price'] = blendstocks.apply(compute_effective_cost, axis=1)

# ------------------------------
# 📉 OPTIMIZATION
# ------------------------------
costs = blendstocks['effective_price'].astype(float).values
n = len(costs)

A_eq = np.ones((1, n))
b_eq = np.array([float(total_volume)])

ethanol_mask = (blendstocks['name'] == 'Ethanol').astype(int).values
A_ub = (-ethanol_mask).reshape(1, n)
b_ub = np.array([-float(total_volume) * float(min_ethanol_ratio)])

bounds = [(0.0, float(total_volume)) for _ in range(n)]

result = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# ------------------------------
# 📊 RESULTS + UX IMPROVEMENTS
# ------------------------------
if result.success:
    blendstocks['blended_volume'] = result.x
    blendstocks['blended_cost'] = blendstocks['blended_volume'] * blendstocks['effective_price']
    total_cost = blendstocks['blended_cost'].sum()
    avg_cost = total_cost / total_volume if total_volume else 0
    ethanol_vol = blendstocks.loc[blendstocks['name']=='Ethanol','blended_volume'].iloc[0]

    # --- Summary Cards ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cost", f"${total_cost:,.2f}")
    c2.metric("Avg Net Cost $/gal", f"${avg_cost:,.3f}")
    c3.metric("Ethanol Vol (gal)", f"{ethanol_vol:,.0f}")

    st.subheader("Optimized Blending Strategy")
    view_cols = ['name', 'base_price', 'effective_price', 'blended_volume', 'blended_cost']
    st.dataframe(blendstocks[view_cols].rename(columns={
        'name':'Component', 'base_price':'Base $/gal', 'effective_price':'Net $/gal',
        'blended_volume':'Gallons to Use', 'blended_cost':'Total $'
    }))

    # Download CSV
    csv = blendstocks[view_cols].to_csv(index=False)
    st.download_button("Download results as CSV", csv, file_name="blend_optimization.csv", mime="text/csv")

    # Charts
    if show_charts:
        vol_chart = alt.Chart(blendstocks).mark_arc().encode(
            theta=alt.Theta(field="blended_volume", type="quantitative"),
            color=alt.Color(field="name", type="nominal"),
            tooltip=["name", "blended_volume"]
        ).properties(title="Blend Volume Share")

        price_chart = alt.Chart(blendstocks).mark_bar().encode(
            x=alt.X('name:N', title='Component'),
            y=alt.Y('effective_price:Q', title='Net $/gal'),
            tooltip=['name','effective_price']
        ).properties(title="Net Cost per Gallon")

        st.altair_chart(vol_chart, use_container_width=True)
        st.altair_chart(price_chart, use_container_width=True)

    # --- Scenario Saver ---
    if 'scenarios' not in st.session_state:
        st.session_state['scenarios'] = []

    if st.button("📌 Save this scenario"):
        st.session_state['scenarios'].append({
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'ethanol_ratio': ethanol_vol/total_volume if total_volume else 0,
            'table': blendstocks[view_cols].copy()
        })
        st.success("Scenario saved. Check below.")

    if st.session_state['scenarios']:
        st.subheader("Saved Scenarios (compare)")
        for i, sc in enumerate(st.session_state['scenarios']):
            with st.expander(f"Scenario {i+1} – {sc['timestamp']}"):
                st.write(f"Total Cost: ${sc['total_cost']:,.2f} | Avg $/gal: ${sc['avg_cost']:,.3f} | Ethanol %: {sc['ethanol_ratio']*100:.1f}%")
                st.dataframe(sc['table'].rename(columns={
                    'name':'Component', 'base_price':'Base $/gal', 'effective_price':'Net $/gal',
                    'blended_volume':'Gallons to Use', 'blended_cost':'Total $'
                }))
else:
    st.error("Optimization failed. Adjust inputs and try again.")

st.caption("If a price couldn't be fetched from EIA, a fallback manual input is used. Click 'Refresh prices' to pull fresh data.")
