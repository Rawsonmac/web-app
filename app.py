# RIN-LCFS Compliance Cost Optimizer – Streamlit App (Non‑linear Specs + Multi‑Objective + Sensitivity)

"""
Major upgrades implemented (items 1 & 2 from your list):
1. **Better chemistry/specs**
   - Optional **non‑linear RVP blend calc** (Antoine-style log mix approximation). Falls back to linear if unchecked.
   - Added **Sulfur, Aromatics, Oxygen, Benzene, CI (gCO2e/MJ)** specs with min/max toggles.
2. **Smarter optimization**
   - **Multi‑objective slider**: minimize Cost vs Carbon Intensity (or BTU) via weighted sum.
   - **Sensitivity/Tornado analysis**: ±10% shocks on key inputs (RINs, LCFS, prices) with chart.
   - **Shadow prices / duals**: show which constraints bind and their marginal cost.

NOTE: Non-linear RVP uses a simple log-sum approximation. For true non-linear mixing, move to Pyomo. This keeps it solvable via LP by linearizing or switching off. Here we solve LP; when non-linear is enabled we do a quick iterative linearization.
"""

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
    Choose single or **multi‑objective** mode and run **sensitivity** instantly.
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

# Physical/Regulatory properties (rough averages) per gallon
DEFAULT_PROPS = pd.DataFrame({
    'name': ["Ethanol","Biodiesel","Renewable Diesel","Gasoline","ULSD"],
    'rvp':  [18.0,0.0,0.0,9.0,0.0],
    'octane':[113.0,0.0,0.0,87.0,0.0],
    'btu':  [76.0,118.0,120.0,114.0,128.0],
    'sulfur_ppm':[0,10,10,30,15],
    'arom_pct':[0,0,0,25,0],
    'oxy_pct':[34,0,0,0,0],
    'benz_pct':[0,0,0,1.0,0],
    'ci_gmj': [60,40,30,93,94]  # gCO2e/MJ; illustrative
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
if st.sidebar.button("↻ Refresh prices"):
    st.cache_data.clear()
    st.experimental_rerun()

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
        mn = st.number_input(f"{comp} min %", value=0.0, min_value=0.0, max_value=1.0, key=f"mn_{comp}")
        mx = st.number_input(f"{comp} max %", value=1.0, min_value=0.0, max_value=1.0, key=f"mx_{comp}")
        comp_bounds[comp]=(mn,mx)

if failures:
    with st.sidebar.expander("API errors"):
        for n,m in failures.items(): st.write(f"**{n}**: {m}")

# ------------------------------
# MAIN INPUTS
# ------------------------------
col1,col2,col3 = st.columns(3)
with col1:
    total_volume = st.number_input("Total Volume (gal)", value=100_000, min_value=0)
with col2:
    min_ethanol_ratio = st.slider("Min Ethanol %", 0.0, 1.0, 0.10)
with col3:
    show_charts = st.checkbox("Charts", True)

st.markdown("### Spec Constraints")
s1,s2,s3 = st.columns(3)
with s1:
    enable_rvp_nl = st.checkbox("Use non-linear RVP", value=False)
    enable_rvp = st.checkbox("Max RVP", value=False)
    max_rvp = st.number_input("Max RVP (psi)", value=9.0, min_value=0.0, disabled=not enable_rvp)
with s2:
    enable_oct = st.checkbox("Min Octane", value=False)
    min_oct = st.number_input("Min Octane", value=87.0, min_value=0.0, disabled=not enable_oct)
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
if enable_oct: add_ge(blendstocks['octane'].values, min_oct*total_volume)
if enable_btu: add_ge(blendstocks['btu'].values,   min_btu*total_volume)
if enable_sulfur: add_le(blendstocks['sulfur_ppm'].values, max_sul*total_volume)
if enable_arom:   add_le(blendstocks['arom_pct'].values,   max_arom*total_volume)
if enable_oxy:    add_le(blendstocks['oxy_pct'].values,    max_oxy*total_volume)
if enable_benz:   add_le(blendstocks['benz_pct'].values,   max_benz*total_volume)
if enable_ci and obj_mode=="Min Cost":  # as hard constraint
    add_le(blendstocks['ci_gmj'].values, max_ci*total_volume)

# Bounds
bounds=[]
for i,row in blendstocks.iterrows():
    mn,mx = comp_bounds[row['name']]
    bounds.append((mn*total_volume, mx*total_volume))

# Stack
A_ub = np.vstack(A_ub_list) if A_ub_list else None
b_ub = np.array(b_ub_list)  if A_ub_list else None

# If non-linear RVP is enabled, do quick iterative linearization
if enable_rvp and enable_rvp_nl:
    # Start with current linear guess
    x_guess = np.array([b[0] for b in bounds])  # min bounds
    x_guess[eth.argmax()] = min_ethanol_ratio*total_volume
    x_guess = x_guess + (total_volume - x_guess.sum())/n
    def rvp_nl(x):
        # log-sum approx: ln(RVPblend) = sum(w_i * ln(RVP_i+1)) ; +1 avoid log 0
        wi = x/ (x.sum()+1e-9)
        return np.exp((wi*np.log(blendstocks['rvp'].values+1)).sum())-1
    for _ in range(3):
        # derivative of rvp wrt x_i approximated by (rvp_nl(x+eps)-rvp)/eps
        base = rvp_nl(x_guess)
        grad = []
        eps = 1e-3*total_volume
        for i in range(n):
            xx = x_guess.copy(); xx[i]+=eps
            grad.append((rvp_nl(xx)-base)/eps)
        grad = np.array(grad)
        # add linear constraint grad·x <= max_rvp - base + grad·x_guess
        A_ub_nl = grad
        b_ub_nl = grad@x_guess + (max_rvp - base)*total_volume/total_volume
        # stack with others
        if A_ub is None:
            A_ub = A_ub_nl.reshape(1,-1); b_ub = np.array([b_ub_nl])
        else:
            A_ub = np.vstack([A_ub, A_ub_nl])
            b_ub = np.hstack([b_ub, b_ub_nl])
        # solve
        res_tmp = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res_tmp.success: break
        x_guess = res_tmp.x

# Solve final LP
res = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res.success:
    blendstocks['blended_volume']=res.x
    # recompute true specs
    tot = total_volume if total_volume else 1
    blendstocks['blended_cost']=blendstocks['blended_volume']*blendstocks['effective_price']
    total_cost = blendstocks['blended_cost'].sum()
    avg_cost = total_cost/tot

    ethanol_vol = blendstocks.loc[blendstocks['name']=='Ethanol','blended_volume'].iloc[0]
    # actual specs
    def vavg(col):
        return float((blendstocks[col]*blendstocks['blended_volume']).sum()/tot)
    act_rvp = vavg('rvp'); act_oct=vavg('octane'); act_btu=vavg('btu')
    act_sul=vavg('sulfur_ppm'); act_ar=vavg('arom_pct'); act_oxy=vavg('oxy_pct'); act_bz=vavg('benz_pct'); act_ci=vavg('ci_gmj')

    # cards
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost", f"${total_cost:,.2f}")
    c2.metric("Avg Net $/gal", f"${avg_cost:,.3f}")
    c3.metric("Ethanol Vol", f"{ethanol_vol:,.0f} gal")
    c4.metric("RVP/Oct/BTU", f"{act_rvp:.1f}/{act_oct:.1f}/{act_btu:.0f}")

    st.caption(f"Specs -> Sulfur {act_sul:.1f} ppm | Arom {act_ar:.1f}% | Oxy {act_oxy:.1f}% | Benz {act_bz:.2f}% | CI {act_ci:.1f} g/MJ")

    view_cols=['name','base_price','effective_price','rvp','octane','btu','sulfur_ppm','arom_pct','oxy_pct','benz_pct','ci_gmj','blended_volume','blended_cost']
    nice = blendstocks[view_cols].rename(columns={'name':'Component','base_price':'Base $/gal','effective_price':'Net $/gal','btu':'BTU','sulfur_ppm':'Sulfur ppm','arom_pct':'Arom %','oxy_pct':'Oxygen %','benz_pct':'Benz %','ci_gmj':'CI g/MJ','blended_volume':'Gallons','blended_cost':'Total $'})
    st.subheader("Optimized Blend")
    st.dataframe(nice)

    csv = nice.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="blend_optimization.csv", mime="text/csv")

    if show_charts:
        vol_chart = alt.Chart(nice).mark_arc().encode(theta=alt.Theta("Gallons:Q"), color="Component:N", tooltip=["Component","Gallons"])
        price_chart = alt.Chart(nice).mark_bar().encode(x="Component:N", y="Net $/gal:Q", tooltip=["Component","Net $/gal"])
        st.altair_chart(vol_chart, use_container_width=True)
        st.altair_chart(price_chart, use_container_width=True)

    # Shadow prices / duals
    st.subheader("Constraint Shadow Prices (marginals)")
    try:
        # SciPy highs returns .ineqlin / .eqlin with marginals
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
        'LCFS':   ('lcfs', None),
        'Gasoline price': ('price','Gasoline'),
        'Ethanol price':  ('price','Ethanol')
    }
    results=[]
    base_total = total_cost
    for lbl,(typ,key) in sens_targets.items():
        for shock in [-0.1,0.1]:
            # clone inputs
            rp = rin_prices.copy(); lp=lcfs_credit_price; fp=final_prices.copy()
            if typ=='rin': rp[key]=rp[key]*(1+shock)
            elif typ=='lcfs': lp = lp*(1+shock)
            elif typ=='price': fp[key]=fp[key]*(1+shock)
            # recompute eff cost only (fast) no re-optimizing full specs to keep speed? we re-run LP quickly
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
                results.append({'Param':lbl, 'Shock':f"{int(shock*100)}%", 'Total Cost':tot2, 'Δ vs Base':tot2-base_total})
    if results:
        df_sens = pd.DataFrame(results)
        st.dataframe(df_sens)
        bar = alt.Chart(df_sens).mark_bar().encode(x='Param:N', y='Δ vs Base:Q', color='Shock:N', tooltip=['Param','Shock','Δ vs Base','Total Cost'])
        st.altair_chart(bar, use_container_width=True)

else:
    st.error("Optimization failed. Relax constraints or adjust bounds.")

st.caption("Non-linear RVP uses log-sum approximation. For exact thermodynamic mixing, switch to a nonlinear solver.")
