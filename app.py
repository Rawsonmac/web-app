# RIN-LCFS Compliance Cost Optimizer – Streamlit App (Specs: RVP / Octane / BTU + UX)

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
    Find the **cheapest blend** (Ethanol, Biodiesel, Renewable Diesel, Gasoline, ULSD) that meets your compliance and spec constraints:
    **RINs / LCFS / Ethanol %, plus RVP, Octane, BTU**. Prices auto-pull from **EIA** where available.
    """
)

# ------------------------------
# 🔑 CONFIG
# ------------------------------
API_KEY = "JM9PgqPjmuvIRmsjQkkwvqk2wcBbowMAF1RLbbhU"  # move to st.secrets["EIA_KEY"] in prod
REFRESH_TTL_SEC = 1800  # 30 min cache

# EIA weekly series IDs ($/gal). Some don't exist -> None and we fallback
EIA_SERIES = {
    "Gasoline": "PET.EMM_EPMRU_PTE_NUS_DPG.W",   # Regular retail gasoline
    "ULSD": "PET.EMD_EPD2D_PTE_NUS_DPG.W",       # Diesel retail price
    "Ethanol": "PET.WEPUPUS3.W",                 # Ethanol FOB
    "Biodiesel": "PET.WEBIOD3.W",                # Biodiesel wholesale (if 404 -> fallback)
    "Renewable Diesel": None                      # No public series -> fallback
}

# Default prices if API fails or missing ($/gal)
DEFAULT_PRICES = {
    "Ethanol": 1.55,
    "Biodiesel": 4.75,
    "Renewable Diesel": 4.10,
    "Gasoline": 2.60,
    "ULSD": 3.00
}

# Approx physical property defaults (simplified, linearized)
# RVP (psi), Octane (R+M/2), BTU per gal (kBTU)
DEFAULT_PROPS = pd.DataFrame({
    'name': ["Ethanol", "Biodiesel", "Renewable Diesel", "Gasoline", "ULSD"],
    'rvp':  [18.0,        0.0,          0.0,                 9.0,        0.0],
    'octane':[113.0,      0.0,          0.0,                 87.0,       0.0],
    'btu':  [76.0,        118.0,        120.0,               114.0,      128.0]
})

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

# Component max/min % sliders (optional)
st.sidebar.subheader("Component Min/Max % (optional)")
comp_bounds = {}
for comp in DEFAULT_PRICES.keys():
    with st.sidebar.expander(comp, expanded=False):
        mn = st.number_input(f"{comp} min %", min_value=0.0, max_value=1.0, value=0.0, key=f"mn_{comp}")
        mx = st.number_input(f"{comp} max %", min_value=0.0, max_value=1.0, value=1.0, key=f"mx_{comp}")
        comp_bounds[comp] = (mn, mx)

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

st.markdown("### Spec Constraints")
sc1, sc2, sc3 = st.columns(3)
with sc1:
    enable_rvp = st.checkbox("Max RVP (psi)", value=False)
    max_rvp = st.number_input("Max RVP", value=9.0, min_value=0.0, disabled=not enable_rvp)
with sc2:
    enable_oct = st.checkbox("Min Octane", value=False)
    min_oct = st.number_input("Min Octane", value=87.0, min_value=0.0, disabled=not enable_oct)
with sc3:
    enable_btu = st.checkbox("Min BTU/gal (kBTU)", value=False)
    min_btu = st.number_input("Min BTU", value=110.0, min_value=0.0, disabled=not enable_btu)

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

# Merge default props
blendstocks = blendstocks.merge(DEFAULT_PROPS, on='name', how='left')

# Effective cost function
def compute_effective_cost(row):
    rin_value = row['rin_yield'] * rin_prices.get(row['rin_type'], 0) if row['rin_type'] else 0.0
    lcfs_value = row['lcfs_credits'] * lcfs_credit_price
    return row['base_price'] - rin_value - lcfs_value

blendstocks['effective_price'] = blendstocks.apply(compute_effective_cost, axis=1)

# ------------------------------
# 📉 OPTIMIZATION SETUP
# ------------------------------
# Objective
costs = blendstocks['effective_price'].astype(float).values
n = len(costs)

# Equality: sum(x_i) = total_volume
A_eq = np.ones((1, n))
b_eq = np.array([float(total_volume)])

# Inequalities list
A_ub_list = []
b_ub_list = []

# Ethanol minimum
eth_mask = (blendstocks['name'] == 'Ethanol').astype(int).values
A_ub_list.append(-eth_mask)  # -x_ethanol <= -min
b_ub_list.append(-float(total_volume) * float(min_ethanol_ratio))

# Spec constraints (linearized as volume weighted)
# RVP: sum(rvp_i * x_i) <= max_rvp * total_volume
if enable_rvp:
    A_ub_list.append(blendstocks['rvp'].fillna(0).astype(float).values)
    b_ub_list.append(max_rvp * float(total_volume))

# Octane: sum(oct_i * x_i) >= min_oct * total_volume  ->  -sum(oct_i*x_i) <= -min_oct*total_volume
if enable_oct:
    A_ub_list.append(-blendstocks['octane'].fillna(0).astype(float).values)
    b_ub_list.append(-min_oct * float(total_volume))

# BTU: sum(btu_i * x_i) >= min_btu * total_volume
if enable_btu:
    A_ub_list.append(-blendstocks['btu'].fillna(0).astype(float).values)
    b_ub_list.append(-min_btu * float(total_volume))

# Component min/max % bounds -> convert to absolute gallons and use bounds tuple
bounds = []
for i, row in blendstocks.iterrows():
    mn_pct, mx_pct = comp_bounds[row['name']]
    low = mn_pct * float(total_volume)
    high = mx_pct * float(total_volume)
    bounds.append((low, high))

# Stack A_ub
if A_ub_list:
    A_ub = np.vstack(A_ub_list)
    b_ub = np.array(b_ub_list)
else:
    A_ub = None
    b_ub = None

# Solve LP
result = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# ------------------------------
# 📊 RESULTS + UX IMPROVEMENTS
# ------------------------------
if result.success:
    blendstocks['blended_volume'] = result.x
    blendstocks['blended_cost'] = blendstocks['blended_volume'] * blendstocks['effective_price']
    total_cost = blendstocks['blended_cost'].sum()
    avg_cost = total_cost / total_volume if total_volume else 0

    ethanol_vol = blendstocks.loc[blendstocks['name']=="Ethanol",'blended_volume'].iloc[0]
    act_rvp = (blendstocks['rvp'] * blendstocks['blended_volume']).sum() / total_volume if total_volume else 0
    act_oct = (blendstocks['octane'] * blendstocks['blended_volume']).sum() / total_volume if total_volume else 0
    act_btu = (blendstocks['btu'] * blendstocks['blended_volume']).sum() / total_volume if total_volume else 0

    # --- Summary Cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost", f"${total_cost:,.2f}")
    c2.metric("Avg Net $/gal", f"${avg_cost:,.3f}")
    c3.metric("Ethanol Vol", f"{ethanol_vol:,.0f} gal")
    c4.metric("RVP / Oct / BTU", f"{act_rvp:.1f} / {act_oct:.1f} / {act_btu:.0f}")

    st.subheader("Optimized Blending Strategy")
    view_cols = ['name','base_price','effective_price','rvp','octane','btu','blended_volume','blended_cost']
    nice = blendstocks[view_cols].rename(columns={
        'name':'Component','base_price':'Base $/gal','effective_price':'Net $/gal',
        'rvp':'RVP','octane':'Octane','btu':'BTU (k/gal)',
        'blended_volume':'Gallons to Use','blended_cost':'Total $'
    })
    st.dataframe(nice)

    # Download CSV
    csv = nice.to_csv(index=False)
    st.download_button("Download results as CSV", csv, file_name="blend_optimization.csv", mime="text/csv")

    # Charts
    if show_charts:
        vol_chart = alt.Chart(nice).mark_arc().encode(
            theta=alt.Theta(field="Gallons to Use", type="quantitative"),
            color=alt.Color(field="Component", type="nominal"),
            tooltip=["Component","Gallons to Use"]
        ).properties(title="Blend Volume Share")

        price_chart = alt.Chart(nice).mark_bar().encode(
            x=alt.X('Component:N'),
            y=alt.Y('Net $/gal:Q'),
            tooltip=['Component','Net $/gal']
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
            'rvp': act_rvp,
            'oct': act_oct,
            'btu': act_btu,
            'table': nice.copy()
        })
        st.success("Scenario saved. Check below.")

    if st.session_state['scenarios']:
        st.subheader("Saved Scenarios (compare)")
        for i, sc in enumerate(st.session_state['scenarios']):
            with st.expander(f"Scenario {i+1} – {sc['timestamp']}"):
                st.write(f"Total Cost: ${sc['total_cost']:,.2f} | Avg $/gal: ${sc['avg_cost']:,.3f} | Ethanol %: {sc['ethanol_ratio']*100:.1f}%")
                st.write(f"Specs: RVP {sc['rvp']:.1f} / Oct {sc['oct']:.1f} / BTU {sc['btu']:.0f}")
                st.dataframe(sc['table'])
else:
    st.error("Optimization failed. Adjust inputs and try again.")

st.caption("If a price couldn't be fetched from EIA, a fallback manual input is used. Click 'Refresh prices' to pull fresh data. All spec blends are treated linearly for speed (approximation).")
