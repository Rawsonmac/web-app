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
    Optimize blend cost with **RIN / LCFS credits** and meet **spec constraints** (RVP, Octane, BTU, Sulfur, Aromatics, Oxygen, Benzene, CI). 
    Choose single or **multi-objective** mode and run **sensitivity** instantly.
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

# Physical/Regulatory properties (includes cetane for diesel blendstocks)
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
    st.experimental_rerun()

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

st.sidebar.subheader("Component Min/Max %")
comp_bounds = {}
for comp in DEFAULT_PRICES.keys():
    with st.sidebar.expander(comp, expanded=False):
        mn = st.number_input(f"{comp} min %", value=0.0 if comp not in ['Ethanol', 'Gasoline'] else 0.1, min_value=0.0, max_value=1.0, key=f"mn_{comp}")
        mx = st.number_input(f"{comp} max %", value=1.0, min_value=0.0, max_value=1.0, key=f"mx_{comp}")
        comp_bounds[comp]=(mn,mx)

if failures:
    with st.sidebar.expander("API errors"):
        for n,m in failures.items(): st.write(f"**{n}**: {m}")

# ------------------------------
# MAIN INPUTS
# ------------------------------
col1,col2,col3,col4 = st.columns(4)
with col1:
    total_volume = st.number_input("Total Volume (gal)", value=100_000, min_value=0)
with col2:
    min_ethanol_ratio = st.slider("Min Ethanol %", 0.0, 1.0, 0.10)
with col3:
    show_charts = st.checkbox("Charts", True)
with col4:
    market_price = st.number_input("Market Price ($/gal)", value=3.00, min_value=0.0)

st.markdown("### Spec Constraints")
s1,s2,s3 = st.columns(3)
with s1:
    enable_rvp_nl = st.checkbox("Use non-linear RVP", value=False)
    enable_rvp = st.checkbox("Max RVP", value=False)
    max_rvp = st.number_input("Max RVP (psi)", value=15.0, min_value=0.0, disabled=not enable_rvp)  # Increased default for high Ethanol
with s2:
    enable_oct = st.checkbox("Min Octane", value=True)
    min_oct = st.number_input("Min Octane", value=90.0, min_value=0.0, disabled=not enable_oct)
    enable_sulfur = st.checkbox("Max Sulfur (ppm)", value=False)
    max_sul = st.number_input("Max Sulfur", value=30.0, min_value=0.0, disabled=not enable_sulfur)
with s3:
    enable_btu = st.checkbox("Min BTU (k/gal)", value=False)
    min_btu = st.number_input("Min BTU", value=110.0, min_value=0.0, disabled=not enable_btu)
    enable_ci = st.checkbox("Max CI (gCO₂e/MJ)", value=False)
    max_ci = st.number_input("Max CI", value=90.0, min_value=0.0, disabled=not enable_ci)

s4,s5 = st.columns(2)
with s4:
    enable_arom = st.checkbox("Max Aromatics %", value=False)
    max_arom = st.number_input("Max Arom %", value=25.0, min_value=0.0, disabled=not enable_arom)
    enable_benz = st.checkbox("Max Benzene %", value=False)
    max_benz = st.number_input("Max Benzene %", value=1.0, min_value=0.0, disabled=not enable_benz)
with s5:
    enable_oxy = st.checkbox("Max Oxygen %", value=False)
    max_oxy = st.number_input("Max Oxygen %", value=10.0, min_value=0.0, disabled=not enable_oxy)

st.markdown("### Objective Mode")
mo1, mo2 = st.columns([2,1])
with mo1:
    obj_mode = st.selectbox("Objective", ["Min Cost","Cost vs CI"], index=0)
with mo2:
    weight_ci = st.slider("Weight CI (0=ignore,1=only CI)", 0.0,1.0,0.3,disabled=(obj_mode=="Min Cost"))

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
missing_props = DEFAULT_PROPS[['rvp', 'octane', 'cetane', 'btu', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci_gmj']].isnull().any()
zero_props = DEFAULT_PROPS[['rvp', 'octane', 'cetane', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci_gmj']].eq(0).any()
if missing_props.any():
    st.warning(f"Missing properties detected: {', '.join(missing_props[missing_props].index)}. Ensure all values are provided in DEFAULT_PROPS.")
if zero_props.any():
    st.info(f"Zero values detected for: {', '.join(zero_props[zero_props].index)}. This is expected for octane (diesel blendstocks) and cetane (gasoline blendstocks). Verify other zeros are intentional.")

# ------------------------------
# BUILD TABLE
# ------------------------------
blendstocks = pd.DataFrame({
    'name': list(DEFAULT_PRICES.keys()),
    'base_price': [final_prices[n] for n in DEFAULT_PRICES.keys()],
    'rin_type': ['D6','D4','D4',None,None],
    'rin_yield':[1.0,1.5,1.7,0.0,0.0],
    'lcfs_credits':[0.5,1.5,1.6,0.0,0.0]
})
blendstocks = blendstocks.merge(DEFAULT_PROPS, on='name', how='left')

# Check for merge issues
if blendstocks[['rvp', 'octane', 'cetane', 'btu', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci_gmj']].isnull().any().any():
    st.error("Merge failed: Some blendstocks lack properties. Ensure all blendstocks in DEFAULT_PRICES have corresponding entries in DEFAULT_PROPS.")
    st.write("Blendstocks DataFrame (debug):", blendstocks)

# Effective cost
def eff_cost(r):
    rinv = r['rin_yield']*rin_prices.get(r['rin_type'],0) if r['rin_type'] else 0
    lcfsv = r['lcfs_credits']*lcfs_credit_price
    return r['base_price'] - rinv - lcfsv

blendstocks['effective_price'] = blendstocks.apply(eff_cost, axis=1)

# Carbon intensity cost term (normalized)
ci_vec = blendstocks['ci_gmj'].fillna(0).astype(float).values
ci_norm = (ci_vec - ci_vec.min())/(ci_vec.max()-ci_vec.min()+1e-9)

# Objective vector
costs = blendstocks['effective_price'].astype(float).values
if obj_mode=="Cost vs CI":
    costs = (1-weight_ci)*costs + weight_ci*ci_norm  # weighted sum

n = len(costs)
A_eq = np.ones((1,n))
b_eq = np.array([float(total_volume)])

# Bounds
bounds=[]
for i,row in blendstocks.iterrows():
    mn,mx = comp_bounds[row['name']]
    bounds.append((mn*total_volume, mx*total_volume))

# Adjust constraints for octane feasibility
if enable_oct:
    max_oct = sum(blendstocks['octane'] * [b[1]/total_volume for b in bounds])
    if min_oct > max_oct:
        st.warning(f"Minimum octane {min_oct} exceeds maximum feasible octane {max_oct:.2f}. Adjusting component bounds and constraints.")
        # Adjust min_ethanol_ratio
        eth_idx = blendstocks['name'] == 'Ethanol'
        min_eth_oct = min_oct / blendstocks.loc[eth_idx, 'octane'].iloc[0] if blendstocks.loc[eth_idx, 'octane'].iloc[0] > 0 else 1.0
        min_ethanol_ratio = max(min_ethanol_ratio, min_eth_oct)
        comp_bounds['Ethanol'] = (min_ethanol_ratio, 1.0)
        # Relax diesel minimums
        for comp in ['Biodiesel', 'Renewable Diesel', 'ULSD']:
            comp_bounds[comp] = (0.0, comp_bounds[comp][1])
        # Update bounds
        bounds = []
        for i,row in blendstocks.iterrows():
            mn,mx = comp_bounds[row['name']]
            bounds.append((mn*total_volume, mx*total_volume))
        # Adjust RVP to accommodate Ethanol
        if enable_rvp:
            max_rvp = max(max_rvp, blendstocks.loc[eth_idx, 'rvp'].iloc[0] * min_ethanol_ratio)
        # Relax other constraints to feasible maximums
        if enable_sulfur:
            max_sul = max(max_sul, (blendstocks['sulfur_ppm'] * [b[1]/total_volume for b in bounds]).sum())
        if enable_arom:
            max_arom = max(max_arom, (blendstocks['arom_pct'] * [b[1]/total_volume for b in bounds]).sum())
        if enable_oxy:
            max_oxy = max(max_oxy, (blendstocks['oxy_pct'] * [b[1]/total_volume for b in bounds]).sum())
        if enable_benz:
            max_benz = max(max_benz, (blendstocks['benz_pct'] * [b[1]/total_volume for b in bounds]).sum())
        if enable_ci and obj_mode=="Min Cost":
            max_ci = max(max_ci, (blendstocks['ci_gmj'] * [b[1]/total_volume for b in bounds]).sum())
        st.info(f"Adjusted constraints: min_ethanol_ratio={min_ethanol_ratio:.2f}, max_rvp={max_rvp:.2f}, max_sul={max_sul:.2f}, max_arom={max_arom:.2f}, max_oxy={max_oxy:.2f}, max_benz={max_benz:.2f}, max_ci={max_ci:.2f}")

# Inequalities
A_ub_list=[]; b_ub_list=[]
# Ethanol min
eth = (blendstocks['name']=='Ethanol').astype(int).values
A_ub_list.append(-eth); b_ub_list.append(-total_volume*min_ethanol_ratio)

# Linear specs
def add_le(vec, limit):
    A_ub_list.append(vec); b_ub_list.append(limit)

def add_ge(vec, limit):
    A_ub_list.append(-vec); b_ub_list.append(-limit)

if enable_rvp and not enable_rvp_nl:
    add_le(blendstocks['rvp'].values, max_rvp*total_volume)
if enable_oct: 
    add_ge(blendstocks['octane'].values, min_oct*total_volume)
if enable_btu: 
    add_ge(blendstocks['btu'].values, min_btu*total_volume)
if enable_sulfur: 
    add_le(blendstocks['sulfur_ppm'].values, max_sul*total_volume)
if enable_arom:   
    add_le(blendstocks['arom_pct'].values, max_arom*total_volume)
if enable_oxy:    
    add_le(blendstocks['oxy_pct'].values, max_oxy*total_volume)
if enable_benz:   
    add_le(blendstocks['benz_pct'].values, max_benz*total_volume)
if enable_ci and obj_mode=="Min Cost":  
    add_le(blendstocks['ci_gmj'].values, max_ci*total_volume)

# Stack
A_ub = np.vstack(A_ub_list) if A_ub_list else None
b_ub = np.array(b_ub_list)  if A_ub_list else None

# If non-linear RVP is enabled, do quick iterative linearization
if enable_rvp and enable_rvp_nl:
    x_guess = np.array([b[0] for b in bounds])
    x_guess[eth.argmax()] = min_ethanol_ratio*total_volume
    x_guess = x_guess + (total_volume - x_guess.sum())/n
    def rvp_nl(x):
        wi = x/ (x.sum()+1e-9)
        return np.exp((wi*np.log(blendstocks['rvp'].values+1)).sum())-1
    for _ in range(3):
        base = rvp_nl(x_guess)
        grad = []
        eps = 1e-3*total_volume
        for i in range(n):
            xx = x_guess.copy(); xx[i]+=eps
            grad.append((rvp_nl(xx)-base)/eps)
        grad = np.array(grad)
        A_ub_nl = grad
        b_ub_nl = grad@x_guess + (max_rvp - base)*total_volume/total_volume
        if A_ub is None:
            A_ub = A_ub_nl.reshape(1,-1); b_ub = np.array([b_ub_nl])
        else:
            A_ub = np.vstack([A_ub, A_ub_nl])
            b_ub = np.hstack([b_ub, b_ub_nl])
        res_tmp = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res_tmp.success: break
        x_guess = res_tmp.x

# Solve final LP
res = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    blendstocks['blended_volume']=res.x
    tot = total_volume if total_volume else 1
    blendstocks['blended_cost']=blendstocks['blended_volume']*blendstocks['effective_price']
    total_cost = blendstocks['blended_cost'].sum()
    avg_cost = total_cost/tot

    ethanol_vol = blendstocks.loc[blendstocks['name']=='Ethanol','blended_volume'].iloc[0]
    def vavg(col):
        return float((blendstocks[col]*blendstocks['blended_volume']).sum()/tot)
    act_rvp = vavg('rvp'); act_oct=vavg('octane'); act_btu=vavg('btu')
    act_sul=vavg('sulfur_ppm'); act_ar=vavg('arom_pct'); act_oxy=vavg('oxy_pct'); act_bz=vavg('benz_pct'); act_ci=vavg('ci_gmj')

    # Diagnostic: Feasible property ranges
    st.subheader("Feasible Property Ranges")
    props = ['rvp', 'octane', 'cetane', 'btu', 'sulfur_ppm', 'arom_pct', 'oxy_pct', 'benz_pct', 'ci_gmj']
    ranges = {}
    for prop in props:
        min_val = (blendstocks[prop] * [b[0]/tot for b in bounds]).sum()
        max_val = (blendstocks[prop] * [b[1]/tot for b in bounds]).sum()
        ranges[prop] = (min_val, max_val)
    st.write(pd.DataFrame(ranges, index=['Min', 'Max']).T)

    # cards
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost", f"${total_cost:,.2f}")
    c2.metric("Avg Net $/gal", f"${avg_cost:,.3f}")
    c3.metric("Ethanol Vol", f"{ethanol_vol:,.0f} gal")
    c4.metric("RVP/Oct/BTU", f"{act_rvp:.1f}/{act_oct:.1f}/{act_btu:.0f}")

    st.caption(f"Specs -> Sulfur {act_sul:.1f} ppm | Arom {act_ar:.1f}% | Oxy {act_oxy:.1f}% | Benz {act_bz:.2f}% | CI {act_ci:.1f} g/MJ")

    # Profit calculation
    st.subheader("Expected Profit")
    revenue = market_price * total_volume
    profit = revenue - total_cost
    profit_per_gal = profit / total_volume if total_volume else 0
    if profit < 0:
        st.warning("Profit is negative. Consider increasing market price or further relaxing constraints.")

    c1, c2 = st.columns(2)
    c1.metric("Total Profit", f"${profit:,.2f}")
    c2.metric("Profit per Gallon", f"${profit_per_gal:,.3f}")

    view_cols=['name','base_price','effective_price','rvp','octane','cetane','btu','sulfur_ppm','arom_pct','oxy_pct','benz_pct','ci_gmj','blended_volume','blended_cost']
    nice = blendstocks[view_cols].rename(columns={
        'name':'Component','base_price':'Base $/gal','effective_price':'Net $/gal','rvp':'RVP (psi)',
        'octane':'Octane','cetane':'Cetane','btu':'BTU (k/gal)','sulfur_ppm':'Sulfur (ppm)',
        'arom_pct':'Aromatics (%)','oxy_pct':'Oxygen (%)','benz_pct':'Benzene (%)','ci_gmj':'CI (gCO₂e/MJ)',
        'blended_volume':'Gallons','blended_cost':'Total $'
    })
    nice = nice.round(2).astype(str).replace('0.0', '0.00')

    # Add summary row
    summary = pd.DataFrame({
        'Component': ['Total'],
        'Base $/gal': [''], 'Net $/gal': [f"{avg_cost:.3f}"], 'RVP (psi)': [f"{act_rvp:.2f}"], 'Octane': [f"{act_oct:.2f}"],
        'Cetane': [''], 'BTU (k/gal)': [f"{act_btu:.2f}"], 'Sulfur (ppm)': [f"{act_sul:.2f}"], 
        'Aromatics (%)': [f"{act_ar:.2f}"], 'Oxygen (%)': [f"{act_oxy:.2f}"], 'Benzene (%)': [f"{act_bz:.2f}"], 
        'CI (gCO₂e/MJ)': [f"{act_ci:.2f}"], 'Gallons': [f"{total_volume:,.0f}"], 'Total $': [f"{total_cost:,.2f}"], 
        'Revenue $': [f"{revenue:,.2f}"], 'Profit $': [f"{profit:,.2f}"]
    })
    nice = pd.concat([nice, summary], ignore_index=True)

    st.subheader("Optimized Blend")
    st.dataframe(nice)

    csv = nice.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="blend_optimization.csv", mime="text/csv")

    if show_charts:
        vol_chart = alt.Chart(nice[:-1]).mark_arc().encode(
            theta=alt.Theta("Gallons:Q"), 
            color="Component:N", 
            tooltip=["Component","Gallons"]
        )
        price_chart = alt.Chart(nice[:-1]).mark_bar(size=40).encode(
            x=alt.X("Component:N", axis=alt.Axis(labelAngle=-15)),
            y=alt.Y("Net $/gal:Q", title="Net $/gal"),
            tooltip=["Component", "Net $/gal"]
        ).properties(height=300)
        st.altair_chart(vol_chart, use_container_width=True)
        st.altair_chart(price_chart, use_container_width=True)

        profit_data = pd.DataFrame({
            'Category': ['Revenue', 'Total Cost', 'Profit'],
            'Value': [revenue, total_cost, profit]
        })
        profit_chart = alt.Chart(profit_data).mark_arc().encode(
            theta=alt.Theta("Value:Q", title="Amount ($)"),
            color=alt.Color("Category:N", scale=alt.Scale(range=["#36A2EB", "#FF6384", "#4CAF50"])),
            tooltip=["Category", alt.Tooltip("Value:Q", format="$,.2f")]
        ).properties(
            title="Profit Breakdown"
        )
        st.altair_chart(profit_chart, use_container_width=True)

    # Shadow prices
    st.subheader("Constraint Shadow Prices (marginals)")
    try:
        m_e = res.eqlin.marginals if hasattr(res, 'eqlin') else []
        m_i = res.ineqlin.marginals if hasattr(res, 'ineqlin') else []
        df_dual = pd.DataFrame({
            'type':['eq']*len(m_e)+['ineq']*len(m_i),
            'marginal':np.concatenate([m_e,m_i])
        })
        st.dataframe(df_dual)
    except Exception:
        st.info("Solver did not return marginals in this SciPy version.")

    # Sensitivity analysis
    st.subheader("Sensitivity (±10% shocks)")
    sens_targets = {
        'D6 RIN': ('rin', 'D6'),
        'D4 RIN': ('rin', 'D4'),
        'LCFS': ('lcfs', None),
        'Gasoline price': ('price','Gasoline'),
        'Ethanol price': ('price','Ethanol'),
        'Market price': ('market', None)
    }
    results=[]
    base_total = total_cost
    base_profit = profit
    for lbl,(typ,key) in sens_targets.items():
        for shock in [-0.1,0.1]:
            rp = rin_prices.copy()
            lp = lcfs_credit_price
            fp = final_prices.copy()
            mp = market_price
            if typ=='rin': rp[key]=rp[key]*(1+shock)
            elif typ=='lcfs': lp = lp*(1+shock)
            elif typ=='price': fp[key]=fp[key]*(1+shock)
            elif typ=='market': mp = mp*(1+shock)
            bs_tmp = blendstocks.copy()
            def ec2(r):
                rinv = r['rin_yield']*rp.get(r['rin_type'],0) if r['rin_type'] else 0
                lcfsv = r['lcfs_credits']*lp
                return fp[r['name']] - rinv - lcfsv
            bs_tmp['effective_price'] = bs_tmp.apply(ec2,axis=1)
            ctmp = bs_tmp['effective_price'].values
            if obj_mode=="Cost vs CI":
                ctmp = (1-weight_ci)*ctmp + weight_ci*ci_norm
            res2 = linprog(c=ctmp, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res2.success:
                tot2 = (res2.x*bs_tmp['effective_price'].values).sum()
                profit2 = (mp * total_volume) - tot2
                results.append({
                    'Param': lbl, 
                    'Shock': f"{int(shock*100)}%", 
                    'Total Cost': tot2, 
                    'Δ Cost vs Base': tot2-base_total,
                    'Profit': profit2,
                    'Δ Profit vs Base': profit2-base_profit
                })
    if results:
        df_sens = pd.DataFrame(results)
        st.dataframe(df_sens)
        bar = alt.Chart(df_sens).mark_bar().encode(
            x='Param:N', 
            y='Δ Profit vs Base:Q', 
            color='Shock:N', 
            tooltip=['Param','Shock','Δ Cost vs Base','Total Cost','Δ Profit vs Base','Profit']
        )
        st.altair_chart(bar, use_container_width=True)

else:
    st.error(f"Optimization failed: {res.message}. Relax constraints further or add high-octane blendstocks (e.g., Premium Gasoline).")
    st.write("Try increasing max_rvp, max_sul, max_arom, max_oxy, max_benz, or max_ci, or adjust component bounds to allow more Ethanol/Gasoline.")

st.caption("Non-linear RVP uses log-sum approximation. For exact thermodynamic mixing, switch to a nonlinear solver.")
