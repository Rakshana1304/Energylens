import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor

st.set_page_config(page_title="EnergyLens | Retail Energy Intelligence",
                   page_icon="⚡", layout="wide")

# ---------------- IKEA-inspired visual system ----------------
st.markdown("""
<style>
:root { --ikea-blue:#0058A3; --ikea-yellow:#FFDA1A; --ink:#111827; --muted:#6B7280; }
.block-container { max-width: 1400px; padding-top: 1.2rem; }
.hero { background:#0058A3; color:white; padding:28px 34px; border-radius:18px; margin-bottom:18px; }
.hero h1 { margin:0; font-size:42px; font-weight:800; }
.hero p { margin:7px 0 0; font-size:17px; opacity:.95; }
.kpi { background:#fff; border:1px solid #E5E7EB; border-radius:16px; padding:18px; min-height:120px; }
.kpi .label { color:#6B7280; font-size:13px; font-weight:700; text-transform:uppercase; }
.kpi .value { color:#111827; font-size:29px; font-weight:800; margin-top:8px; }
.kpi .sub { color:#6B7280; font-size:12px; margin-top:4px; }
.action { border-left:6px solid #FFDA1A; background:#FFFBE6; padding:15px 18px; border-radius:10px; margin:8px 0; }
.badge { display:inline-block; padding:5px 10px; border-radius:999px; background:#F3F4F6; font-size:12px; font-weight:700; }
.small { color:#6B7280; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Core logic from uploaded EnergyLens ----------------
STORES = {"Store A": 20000, "Store B": 15000, "Store C": 18000}
EMISSION_FACTOR = 0.71
TARIFF = 8.0
SOLAR_KWP = 600
PR_EXPECTED = 0.80
PR_ALERT = 0.75

@st.cache_data
def make_simulated_data():
    np.random.seed(42)
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h")
    def make_store(name, area, efficiency):
        hour, dow, doy = idx.hour, idx.dayofweek, idx.dayofyear
        temp = (28 + 5*np.sin(2*np.pi*(doy-100)/365)
                + 3*np.sin(2*np.pi*(hour-14)/24)
                + np.random.normal(0,1,len(idx)))
        is_open = (hour >= 10) & (hour < 22)
        base_kw = area * 0.030 * efficiency
        load = base_kw * (0.35 + 0.65*is_open)
        load = load * (1 + 0.015*(temp-28))
        load = load * np.where(dow >= 5, 1.10, 1.0)
        load = load * np.random.normal(1,0.03,len(idx))
        return pd.DataFrame({"store":name, "temp_c":temp, "kwh":load}, index=idx)

    frames = []
    for (name, area), eff in zip(STORES.items(), [0.95,1.00,1.05]):
        frames.append(make_store(name, area, eff))
    raw = pd.concat(frames)
    raw.index.name = "timestamp"

    # Same intentional problems as the uploaded script
    m=(raw.store=="Store B")&(raw.index>="2025-06-01")&(raw.index<"2025-07-16")&(raw.index.hour<6)
    raw.loc[m,"kwh"] *= 2.0
    m=(raw.store=="Store C")&(raw.index>="2025-08-10")&(raw.index<"2025-08-15")
    raw.loc[m,"kwh"] = raw.loc[m,"kwh"].iloc[0]
    m=(raw.store=="Store C")&(raw.index=="2025-03-15 14:00")
    raw.loc[m,"kwh"] *= 6
    m=(raw.store=="Store A")&(raw.index>="2025-10-03")&(raw.index<"2025-10-04 16:00")
    raw.loc[m,"kwh"] = np.nan

    hour, doy = np.asarray(idx.hour), np.asarray(idx.dayofyear)
    season=0.85+0.15*np.cos(2*np.pi*(doy-80)/365)
    irr=(np.clip(np.sin(np.pi*(hour-6)/12),0,None)*season*
         np.clip(np.random.normal(0.9,0.12,len(idx)),0.3,1.0))
    gen=SOLAR_KWP*irr*PR_EXPECTED
    gen[(idx>="2025-03-01")&(idx<"2025-05-01")]*=0.82
    solar=pd.DataFrame({"irradiation":irr,"gen_kwh":gen},index=idx)
    return raw, solar

def prepare(raw):
    df=raw.copy()
    df["hour"]=df.index.hour
    df["dow"]=df.index.dayofweek
    df["month"]=df.index.month
    df["issue"]=""
    quality=[]
    for s,g in df.groupby("store"):
        missing=g["kwh"].isna()
        dup=g.index.duplicated().sum()
        same=g["kwh"].diff()==0
        run=same.groupby((~same).cumsum()).cumsum()
        flat=run>=5
        sel=(df.store==s).values
        df.loc[sel,"issue"]=np.where(missing.values,"missing",np.where(flat.values,"flat-line",""))
        quality.append({"store":s,"missing_hours":int(missing.sum()),
                        "duplicate_rows":int(dup),"flat_line_hours":int(flat.sum())})
    quality=pd.DataFrame(quality)
    bad=df.issue!=""
    df["kwh_clean"]=df.kwh.where(~bad)
    typical=df.groupby(["store","dow","hour"])["kwh_clean"].transform("median")
    df["kwh_clean"]=df.kwh_clean.fillna(typical)
    df["baseline"]=df.groupby(["store","dow","hour"])["kwh_clean"].transform("median")
    df["ratio"]=df.kwh_clean/df.baseline
    df["rule_flag"]=df.ratio>1.4
    df["iso_flag"]=False
    for s in STORES:
        ix=df.store==s
        X=df.loc[ix,["ratio","hour","temp_c"]]
        model=IsolationForest(contamination=0.01,random_state=42).fit(X)
        df.loc[ix,"iso_flag"]=model.predict(X)==-1
    df["excess_kwh"]=np.where(df.rule_flag,df.kwh_clean-df.baseline,0)
    return df,quality

def metrics(df):
    kpi=df.groupby("store")["kwh_clean"].sum().to_frame("annual_kwh")
    kpi["area_m2"]=pd.Series(STORES)
    kpi["EUI_kWh_per_m2"]=(kpi.annual_kwh/kpi.area_m2).round(1)
    kpi["CO2_tonnes"]=(kpi.annual_kwh*EMISSION_FACTOR/1000).round(1)
    kpi["cost_INR"]=kpi.annual_kwh*TARIFF
    sav=df.groupby("store")["excess_kwh"].sum().to_frame("excess_kwh")
    sav["cost_INR"]=sav.excess_kwh*TARIFF
    sav["CO2_tonnes"]=sav.excess_kwh*EMISSION_FACTOR/1000
    return kpi,sav

def forecast_monthly(df, store, months=3):
    monthly=df[df.store==store].set_index(df[df.store==store].index).resample("MS").kwh_clean.sum().reset_index()
    monthly.columns=["date","kwh"]
    monthly["t"]=np.arange(len(monthly))
    if len(monthly)<6:
        return pd.DataFrame()
    model=RandomForestRegressor(n_estimators=250,random_state=42,min_samples_leaf=2)
    model.fit(monthly[["t"]],monthly["kwh"])
    future_t=np.arange(len(monthly),len(monthly)+months)
    future_dates=pd.date_range(monthly.date.max()+pd.offsets.MonthBegin(1),periods=months,freq="MS")
    pred=model.predict(future_t.reshape(-1,1))
    future=pd.DataFrame({"date":future_dates,"kwh":pred,"type":"Forecast"})
    historical=monthly.assign(type="Historical")
    return pd.concat([historical,future],ignore_index=True)

raw, solar = make_simulated_data()

# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <h1>⚡ EnergyLens</h1>
  <p>Retail Store Energy Monitoring & Savings Analyzer</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("EnergyLens Controls")
uploaded=st.sidebar.file_uploader("Upload meter CSV", type=["csv"])
st.sidebar.caption("Expected columns: timestamp, store, kwh, temp_c")
use_upload=uploaded is not None

if use_upload:
    try:
        u=pd.read_csv(uploaded)
        required={"timestamp","store","kwh"}
        if not required.issubset(u.columns):
            st.sidebar.error("CSV needs timestamp, store and kwh columns.")
            raw_used=raw
        else:
            u["timestamp"]=pd.to_datetime(u["timestamp"])
            raw_used=u.set_index("timestamp").sort_index()
            if "temp_c" not in raw_used.columns:
                raw_used["temp_c"]=28.0
            st.sidebar.success("Uploaded dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")
        raw_used=raw
else:
    raw_used=raw

df,quality=prepare(raw_used)
available=sorted(df.store.unique())
selected=st.sidebar.multiselect("Stores",available,default=available)
df=df[df.store.isin(selected)]

if df.empty:
    st.warning("Select at least one store with data to view the dashboard.")
    st.stop()

start,end=df.index.min().date(),df.index.max().date()
dates=st.sidebar.date_input("Date range",(start,end))
if isinstance(dates,tuple) and len(dates)==2:
    df=df[(df.index.date>=dates[0])&(df.index.date<=dates[1])]

if df.empty:
    st.warning("No data is available for the selected date range.")
    st.stop()

kpi,sav=metrics(df)
anomaly_hours=int(df.rule_flag.sum())
energy=df.kwh_clean.sum()
cost=energy*TARIFF
co2=energy*EMISSION_FACTOR/1000
solar_total=solar.gen_kwh.sum()

# ---------------- KPI row ----------------
cols=st.columns(5)
items=[
    ("Electricity",f"{energy/1e6:.2f} GWh","cleaned meter consumption"),
    ("Energy cost",f"₹{cost/1e7:.2f} Cr","demo tariff assumption"),
    ("CO₂ emissions",f"{co2:,.0f} t","grid-factor estimate"),
    ("Anomalies",f"{anomaly_hours:,}","rule-based flags"),
    ("Solar",f"{solar_total/1e6:.2f} GWh","simulated PV generation"),
]
for c,(label,value,sub) in zip(cols,items):
    c.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)

st.write("")
tab1,tab2,tab3,tab4,tab5=st.tabs(["Overview","Anomalies & Actions","Forecasting","Solar PV","Data Quality"])

with tab1:
    st.subheader("Energy performance")
    daily=df.resample("D").kwh_clean.sum().reset_index()
    fig=px.line(daily,x="timestamp",y="kwh_clean",title="Daily electricity consumption")
    fig.update_layout(height=400,margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        b=df.groupby("store").kwh_clean.sum().reset_index()
        fig=px.bar(b,x="store",y="kwh_clean",title="Annual electricity by store")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        e=kpi.reset_index()
        fig=px.bar(e,x="store",y="EUI_kWh_per_m2",title="Energy Use Intensity (kWh/m²/year)")
        st.plotly_chart(fig,use_container_width=True)

with tab2:
    st.subheader("AI-assisted anomaly detection")
    a=df[df.rule_flag | df.iso_flag].copy()
    a["severity"]=np.where(a.rule_flag & a.iso_flag,"High","Review")
    st.dataframe(a.reset_index()[["timestamp","store","kwh_clean","baseline","ratio","rule_flag","iso_flag","severity"]].sort_values("ratio",ascending=False).head(100),
                 use_container_width=True)
    st.markdown("### Recommended actions")
    actions=[]
    night=df[(df.store=="Store B")&df.rule_flag&(df.hour<6)]
    if len(night): actions.append("Store B: investigate HVAC/lighting schedules in the BMS during night hours.")
    spike=df[(df.store=="Store C")&(df.ratio>3)]
    if len(spike): actions.append("Store C: investigate the one-hour consumption spike for a meter or equipment event.")
    if (quality.flat_line_hours>0).any(): actions.append("Check flat-lined meters for communication loss or meter faults.")
    if (quality.missing_hours>0).any(): actions.append("Check Store A meter communication/logging for the missing-data period.")
    if not actions: actions=["No rule-based action was triggered for the selected period."]
    for x in actions: st.markdown(f'<div class="action">💡 {x}</div>',unsafe_allow_html=True)

with tab3:
    st.subheader("Energy consumption forecasting")
    store=st.selectbox("Store for forecast",available)
    horizon=st.slider("Forecast horizon (months)",1,6,3)
    fc=forecast_monthly(df,store,horizon)
    if fc.empty:
        st.warning("Not enough historical data for forecasting.")
    else:
        fig=px.line(fc,x="date",y="kwh",color="type",markers=True,
                    title=f"{store}: historical consumption and forecast")
        fig.update_layout(height=430)
        st.plotly_chart(fig,use_container_width=True)
        future=fc[fc.type=="Forecast"]
        st.dataframe(future.rename(columns={"date":"Month","kwh":"Forecast kWh"}),use_container_width=True)
        st.caption("Forecast uses a Random Forest regression on historical monthly consumption. It is a portfolio-demo model, not a production forecasting model.")

with tab4:
    st.subheader("Solar PV performance")
    sm=solar.resample("MS").sum()
    sm["PR"]=sm.gen_kwh/(SOLAR_KWP*sm.irradiation)
    sm["lost_kwh"]=np.where(sm.PR<PR_ALERT,
                            SOLAR_KWP*sm.irradiation*PR_EXPECTED-sm.gen_kwh,0)
    fig=px.line(sm.reset_index(),x="index",y="PR",markers=True,
                title="Monthly Solar PV Performance Ratio")
    fig.add_hline(y=PR_ALERT,line_dash="dash",annotation_text="Alert threshold")
    st.plotly_chart(fig,use_container_width=True)
    low=sm[sm.PR<PR_ALERT]
    if len(low):
        st.warning("Low-performance months detected. Investigate soiling, shading, inverter or availability issues.")
    c1,c2=st.columns(2)
    c1.metric("Average PR",f"{sm.PR.mean():.2f}")
    c2.metric("Estimated solar loss",f"{sm.lost_kwh.sum():,.0f} kWh")

with tab5:
    st.subheader("Data quality and integrity")
    st.dataframe(quality,use_container_width=True)
    st.write("Issue handling: missing/flat-line readings are excluded and replaced with the store's typical hour-of-week median before KPI analysis.")
    csv=df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download analyzed dataset",csv,"energylens_analyzed.csv","text/csv")
    st.download_button("Download KPI summary",kpi.reset_index().to_csv(index=False).encode("utf-8"),
                       "energylens_kpi.csv","text/csv")

st.divider()
st.caption("Portfolio demo. All default data is simulated. Replace tariff, emission factor, store area and solar assumptions with verified organizational data before real-world use.")
