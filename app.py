import pandas as pd
import numpy as np
from scipy.optimize import linprog
import streamlit as st
import requests
from datetime import datetime
import altair as alt

st.set_page_config(page_title="RIN / LCFS Optimizer", layout="wide")

st.title("OIL BROKERAGE")
st.markdown(
    """
    Enter a target octane and total volume to optimize a gasoline blend at the **lowest cost** with **RIN / LCFS credits**. 
    Other constraints are automatically adjusted to ensure feasibility.
    """
)

# ------------------------------
# 🔑 CONFIG
# ------------------------------
API_KEY = "JM9PgqPjmuvIRmsjQkkwvqk2wcBbowMAF1RLbbhU"  # move to st.secrets in prod
REFRESH_TTL_SEC = 1800

EIA_SERIES = {
    "Gasoline": "PET.EMM_EPMRU_PTE_NUS_DPG.W",
    "ULSD": "PET.EMD_EPD2D_PTE_NUS_DPG.W",
    "Ethanol": "PET.WEPUPUS3.W",
    "Biodiesel": "PET.WEBIOD3.W",
    "Renewable Diesel": None
}

DEFAULT_PRICES = {
    "Ethanol": 1.55,
    "Biodiesel": 4.75,
    "Renewable Diesel": 4.10,
    "Gasoline": 2.60,
    "ULSD": 3.00
}

# Physical/Regulatory properties
DEFAULT_PROPS = pd.DataFrame({
    'name': ["Ethanol", "Biodiesel", "Renewable Diesel", "Gasoline", "ULSD"],
    'rvp': [18.0, 0.0, 0.0, 9.0, 0.0],
    'octane': [113.0, 0.0, 0.0, 87.0, 0.0],
    'cetane': [0.0, 55.0, 80.0, 0.0, 45.0],
    'btu': [76.0, 118.0, 120.0, 114.0, 128.0],
    'sulfur_ppm': [0.0, 15.0, 10.0, 20.0, 15.0],
    'arom_pct': [0.0, 0.0, 5.0, 22.5, 15.0],
    'oxy_pct': [34.7, 11.0, 0.0, 3.7, 0.0],
    'benz_pct': [0.0, 0.0, 0.0, 0.6, 0.0],
    'ci_gmj': [60.0, 40.0, 30.0, 93.0, 94.0]
})

DEFAULT_RIN_PRICES = {"D6":0.83,"D4":1.17,"D5":1.05,"D3":2.48}
DEFAULT_LCFS_PRICE = 92.5

# ------------------------------
# 🛰️ DATA FETCH
# ------------------------------
@st.cache_data(ttl=REFRESH_TTL_SEC)
def fetch_eia_price(series_id, api_key):
    if series_id is None:
        return None
    url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    js = r.json()
    return float(js["series"][0]["data"][0][1])

@st.cache_data(ttl=REFRESH_TTL_SEC)
def get_live_prices():
    prices, fails = {}, {}
    for n,s in EIA_SERIES.items():
        try:
            prices[n] = fetch_eia_price(s, API_KEY)
        except Exception as e:
            prices[n] = None
            fails[n] = str(e)
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    return prices, fails, ts

# ------------------------------
# SIDEBAR INPUTS
# ------------------------------
st.sidebar.header("Data & Inputs")
if "refresh_prices" not in st.session_state:
    st.session_state.refresh_prices = False

if st.sidebar.button("↻ Refresh prices"):
    st.cache_data.clear()
    st.session_state.refresh_prices = True
    st.rerun()  # Updated from st.experimental_rerun()

if st.session_state.refresh_prices:
    st.session_state.refresh_prices = False

blend_prices, failures, fetched_at = get_live_prices()
rin_prices = DEFAULT_RIN_PRICES.copy()
lcfs_credit_price = DEFAULT_LCFS_PRICE
st.sidebar.caption(f"Fetched: {fetched_at} (cache {REFRESH_TTL_SEC//60}m)")

st.sidebar.subheader("RIN Prices ($/RIN)")
for k,v in rin_prices.items():
    rin_prices[k] = st.sidebar.number_input(f"{k}", value=float(v))

lcfs_credit_price = st.sidebar.number_input("LCFS $/MT CO₂", value=float(lcfs_credit_price))

st.sidebar.subheader("Blendstock Prices ($/gal)")
final_prices = {}
for n,dv in DEFAULT_PRICES.items():
    lv = blend_prices.get(n)
    tag = "live" if lv is not None else "fallback"
    final_prices[n] = st.sidebar.number_input(f"{n} ({tag})", value=float(lv or dv))

if failures:
    with st.sidebar.expander("API errors"):
        for n,m in failures.items(): st.write(f"**{n}**: {m}")

# Debug toggle
st.sidebar.subheader("Debug Options")
show_debug = st.sidebar.checkbox("Show diagnostic messages", value=False)

# ------------------------------
# MAIN INPUTS
# ------------------------------
st.markdown("### Target Specifications")
col1,col2,col3 = st.columns(3)
with col1:
    total_volume = st.number_input("Total Volume (gal)", value=100_000, min_value=0)
with col2:
    min_oct = st.number_input("Target Octane", value=91.0, min_value=0.0)
with col3:
    market_price = st.number_input("Market Price ($/gal)", value=3.00, min_value=0.0)

# Optional advanced constraints
with st.expander("Advanced Constraints (optional)", expanded=False):
    enable_rvp = st.checkbox("Max RVP", value=False)
    max_rvp = st.number_input("Max RVP (psi)", value=15.0, min_value=0.0, disabled=not enable_rvp)
    enable_sulfur = st.checkbox("Max Sulfur (ppm)", value=False)
    max_sul = st.number_input("Max Sulfur", value=30.0, min_value=0.0, disabled=not enable_sulfur)
    enable_btu = st.checkbox("Min BTU (k/gal)", value=False)
    min_btu = st.number_input("Min BTU", value=110.0, min_value=0.0, disabled=not enable_btu)
    enable_ci = st.checkbox("Max CI (gCO₂e/MJ)", value=False)
    max_ci = st.number_input("Max CI", value=90.0, min_value=0.0, disabled=not enable_ci)
    enable_arom = st.checkbox("Max Aromatics %", value=False)
    max_arom = st.number_input("Max Arom %", value=25.0, min_value=0.0, disabled=not enable_arom)
    enable_oxy = st.checkbox("Max Oxygen %", value=False)
    max_oxy = st.number_input("Max Oxygen %", value=10.0, min_value=0.0, disabled=not enable_oxy)
    enable_benz = st.checkbox("Max Benzene %", value=False)
    max_benz = st.number_input("Max Benzene %", value=1.0, min_value=0.0, disabled=not enable_benz)

# ------------------------------
# DISPLAY FEEDSTOCK PROPERTIES
# ------------------------------
st.subheader("Feedstock Properties")
st.caption("Note: Biodiesel, Renewable Diesel, and ULSD have octane=0.00 as they are diesel fuels, not used for spark-ignition engines. Cetane is shown for diesel blendstocks but does not affect octane or gasoline-focused optimization.")
props_display = DEFAULT_PROPS[['name', 'rvp', 'octane', 'cetane', 'btu', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci_gmj']].rename(columns={
    'name': 'Component', 'rvp': 'RVP (psi)', 'octane': 'Octane', 'cetane': 'Cetane',
    'btu': 'BTU (k/gal)', 'sulfur_ppm': 'Sulfur (ppm)', 'arom_pct': 'Aromatics (%)',
    'oxy_pct': 'Oxygen (%)', 'benz_pct': 'Benzene (%)', 'ci_gmj': 'CI (gCO₂e/MJ)'
})
props_display = props_display.round(2).astype(str).replace('0.0', '0.00')
st.dataframe(props_display)

# Validate feedstock properties
if 'diagnostics' not in st.session_state:
    st.session_state.diagnostics = []

missing_props = DEFAULT_PROPS[['rvp', 'octane', 'cetane', 'btu', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci_gmj']].isnull().any()
zero_props = DEFAULT_PROPS[['rvp', 'octane', 'cetane', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci
