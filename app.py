from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor

# ============================================================
# PriceLens — ML-assisted price anomaly exploration
# ============================================================

st.set_page_config(
    page_title="PriceLens",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "price_expectation_model.cbm"
GROUP_PATH = DATA_DIR / "Price_Fairness_Lens_group_scores_final.csv"
CONTEXT_PATH = DATA_DIR / "Price_Fairness_Lens_context_scores.csv"
MODEL_DATA_PATH = DATA_DIR / "Consumer_Reports_model_data_repaired.csv"

MODEL_FEATURES = [
    "Retailer", "Store", "Date", "Hour", "Product_Number",
    "Product_Name", "Original_Price", "Promotion"
]
CAT_FEATURES = [
    "Retailer", "Store", "Date", "Product_Number",
    "Product_Name", "Promotion"
]

# ------------------------------------------------------------
# Visual styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { background: #f7f8fa; }
    .block-container { max-width: 1280px; padding-top: 1rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: Manrope, Inter, sans-serif !important; letter-spacing: -.03em; }

    .brand { font-family: Manrope, sans-serif; font-size: 2rem; font-weight: 800;
             letter-spacing: -.05em; color: #15171b; }
    .tagline { color: #747a84; font-size: .82rem; margin-top: -.15rem; }
    .hero { background: linear-gradient(135deg,#fff,#eef1f4); border: 1px solid #e2e5e9;
            border-radius: 24px; padding: 3rem 2.2rem; margin: 1rem 0 1.5rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem;
               font-weight: 700; color: #6e7580; }
    .hero-title { font-family: Manrope, sans-serif; font-size: 3.15rem; font-weight: 800;
                  line-height: 1.05; letter-spacing: -.055em; color: #15171b; max-width: 800px; }
    .hero-copy { color: #626873; font-size: 1rem; line-height: 1.65; max-width: 700px; margin-top: .9rem; }
    .card { background: #fff; border: 1px solid #e2e5e9; border-radius: 18px; padding: 1.2rem 1.3rem;
            box-shadow: 0 3px 18px rgba(20,24,32,.035); margin-bottom: 1rem; }
    .label { text-transform: uppercase; letter-spacing: .1em; font-size: .7rem; color: #777e88; font-weight: 700; }
    .big { font-family: Manrope, sans-serif; font-size: 2rem; font-weight: 800; line-height: 1.1; color: #181b20; }
    .result-price { font-family: Manrope, sans-serif; font-size: 3rem; font-weight: 800; letter-spacing: -.05em; }
    .muted { color: #737984; font-size: .88rem; line-height: 1.55; }
    .pill { display:inline-block; padding:.42rem .75rem; border-radius:999px; border:1px solid #dfe3e7;
            background:#f3f5f7; color:#30343a; font-size:.75rem; font-weight:700; letter-spacing:.03em; }
    .notice { background:#f2f4f6; border-left:3px solid #616973; border-radius:10px; padding:.9rem 1rem;
              color:#505660; font-size:.86rem; line-height:1.55; }
    .footer { color:#8b9199; text-align:center; font-size:.75rem; padding-top:2rem; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e5e9; border-radius:16px; padding:1rem; }
    button { border-radius:12px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

@st.cache_data

def load_data():
    for path in [GROUP_PATH, CONTEXT_PATH, MODEL_DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    group = pd.read_csv(GROUP_PATH)
    context = pd.read_csv(CONTEXT_PATH)
    model_data = pd.read_csv(MODEL_DATA_PATH)

    numeric_group = [
        "Actual_Price", "Peer_Group_Count", "Peer_Group_Unique_Prices",
        "Peer_Group_Min", "Peer_Group_Median", "Peer_Group_Max",
        "Peer_Deviation_Pct", "Peer_Max_Min_Spread_Pct", "Peer_Price_Position",
        "OOF_Predicted_Price", "OOF_Model_Deviation_Pct", "Evidence_Score",
        "Group_Respondents"
    ]
    for col in numeric_group:
        if col in group.columns:
            group[col] = pd.to_numeric(group[col], errors="coerce")

    for col in ["Hour", "Original_Price", "Price_Observed"]:
        if col in model_data.columns:
            model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    return group, context, model_data


@st.cache_resource

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    return model


try:
    group_df, context_df, model_df = load_data()
    model = load_model()
except Exception as exc:
    st.error("PriceLens could not load its model/data files.")
    st.code(str(exc))
    st.stop()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_date(value):
    d = pd.to_datetime(value, errors="coerce")
    return d.strftime("%Y-%m-%d") if pd.notna(d) else str(value)


def get_peers(context_id):
    return group_df[group_df["Context_ID"].astype(str) == str(context_id)].copy().sort_values("Actual_Price")


def make_model_input(context_id):
    rows = model_df[model_df["Context_ID"].astype(str) == str(context_id)].copy()
    if rows.empty:
        raise ValueError("No model features found for this pricing context.")

    row = rows.iloc[0]
    x = pd.DataFrame([{
        "Retailer": row.get("Retailer"),
        "Store": row.get("Store"),
        "Date": normalize_date(row.get("Date")),
        "Hour": row.get("Hour"),
        "Product_Number": row.get("Product_Number"),
        "Product_Name": row.get("Product_Name"),
        "Original_Price": row.get("Original_Price"),
        "Promotion": row.get("Promotion"),
    }])

    for col in CAT_FEATURES:
        x[col] = x[col].fillna("Unknown").astype(str)
    x["Hour"] = pd.to_numeric(x["Hour"], errors="coerce").fillna(0)
    x["Original_Price"] = pd.to_numeric(x["Original_Price"], errors="coerce")
    med = pd.to_numeric(model_df["Original_Price"], errors="coerce").median()
    x["Original_Price"] = x["Original_Price"].fillna(0 if pd.isna(med) else med)
    return x[MODEL_FEATURES]


def analyze_price(price, peers):
    p = peers["Actual_Price"].dropna().astype(float)
    if p.empty:
        return None

    median = float(p.median())
    minimum = float(p.min())
    maximum = float(p.max())
    count = int(p.size)
    unique = int(p.nunique())
    deviation = ((float(price) / median) - 1) * 100 if median > 0 else 0
    spread = ((maximum / minimum) - 1) * 100 if minimum > 0 else 0
    evidence = float(np.clip(max(deviation, 0) / 25 * 100, 0, 100))

    if count < 3:
        signal = "Insufficient evidence"
    elif deviation <= 5:
        signal = "Low signal"
    elif deviation <= 10:
        signal = "Moderate signal"
    elif deviation <= 20:
        signal = "High signal"
    else:
        signal = "Very high signal"

    if deviation <= 2:
        level = "Within observed peer range"
    elif deviation <= 5:
        level = "Slightly above peers"
    elif deviation <= 10:
        level = "Noticeably above peers"
    elif deviation <= 20:
        level = "Substantially above peers"
    else:
        level = "Strong price anomaly signal"

    if price < minimum:
        position = "Below observed range"
    elif price > maximum:
        position = "Above observed range"
    else:
        position = "Within observed range"

    return {
        "price": float(price), "median": median, "min": minimum, "max": maximum,
        "count": count, "unique": unique, "deviation": deviation, "spread": spread,
        "evidence": evidence, "signal": signal, "level": level, "position": position
    }


def context_label(row):
    date = pd.to_datetime(row.get("Date"), errors="coerce")
    date_text = date.strftime("%d %b %Y") if pd.notna(date) else str(row.get("Date", ""))
    try:
        hour = int(float(row.get("Hour", 0)))
        time_text = f"{hour:02d}:00"
    except Exception:
        time_text = str(row.get("Hour", ""))
    return f"{row['Product_Name']}  |  {row['Retailer']} / {row['Store']}  |  {date_text} {time_text}  |  Context {row['Context_ID']}"


def signal_pill(signal):
    return f'<span class="pill">{signal.upper()}</span>'


def peer_chart(peers, selected_price):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=peers["Actual_Price"],
        y=peers["Group"].astype(str),
        mode="markers",
        marker=dict(size=13),
        hovertemplate="$%{x:.2f}<extra></extra>",
        name="Observed peer prices",
    ))
    fig.add_vline(x=float(selected_price), line_dash="dash", annotation_text="Selected price")
    fig.update_layout(
        template="plotly_white", height=350, margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Price ($)", yaxis_title="Pricing group", showlegend=False
    )
    return fig


def add_history(record):
    history = st.session_state.setdefault("history", [])
    if history and history[0] == record:
        return
    history.insert(0, record)
    st.session_state.history = history[:20]


def go(page):
    st.session_state.nav = page

# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------

nav_col, menu_col = st.columns([1.5, 8.5])
with nav_col:
    st.markdown('<div class="brand">PriceLens</div><div class="tagline">See beyond the price tag.</div>', unsafe_allow_html=True)
with menu_col:
    pages = ["Home", "Check Price", "Compare", "Explore", "Insights", "History", "How It Works"]
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"
    st.radio("Navigation", pages, horizontal=True, key="nav", label_visibility="collapsed")
page = st.session_state.nav

# ============================================================
# HOME
# ============================================================
if page == "Home":
    st.markdown(
        '<div class="hero"><div class="eyebrow">Price intelligence for consumers</div>'
        '<div class="hero-title">Is the price you are seeing unusual?</div>'
        '<div class="hero-copy">PriceLens compares a selected grocery price with comparable observed prices and uses a machine-learning model to estimate an expected price.</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Check a Price", type="primary"):
        go("Check Price")
        st.rerun()

    total_contexts = int(context_df["Context_ID"].nunique())
    varying = int((context_df["Peer_Unique_Prices"] > 1).sum())
    groups = int(group_df[["Context_ID", "Group"]].drop_duplicates().shape[0])
    spread = float(context_df["Peer_Max_Min_Spread_Pct"].max())

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pricing contexts", total_contexts)
    c2.metric("Pricing groups", groups)
    c3.metric("Contexts with variation", varying)
    c4.metric("Largest observed spread", f"{spread:.2f}%")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><div class="label">What PriceLens does</div><h3>Compare, predict, explain.</h3><div class="muted">Direct peer-price comparison is the primary evidence layer. The CatBoost model provides a supporting expected-price estimate.</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="label">Important distinction</div><h3>A signal is not a verdict.</h3><div class="muted">An unusual observed price can have multiple explanations. PriceLens does not establish personalized pricing, discrimination, or intent.</div></div>', unsafe_allow_html=True)

# ============================================================
# CHECK PRICE
# ============================================================
elif page == "Check Price":
    st.title("Check a Price")
    st.caption("Choose a comparable context, enter the price you see, and review the evidence behind the signal.")

    products = sorted(context_df["Product_Name"].dropna().astype(str).unique())
    product = st.selectbox("Product", products)

    f = context_df[context_df["Product_Name"].astype(str) == product].copy()
    retailers = sorted(f["Retailer"].dropna().astype(str).unique())
    retailer = st.selectbox("Retailer", retailers)

    f = f[f["Retailer"].astype(str) == retailer]
    stores = sorted(f["Store"].dropna().astype(str).unique())
    store = st.selectbox("Store", stores)

    f = f[f["Store"].astype(str) == store].copy()
    f["Context_Label"] = f.apply(context_label, axis=1)
    cmap = dict(zip(f["Context_Label"], f["Context_ID"]))
    selected_context_label = st.selectbox("Pricing context", list(cmap))
    context_id = cmap[selected_context_label]
    peers = get_peers(context_id)

    if peers.empty:
        st.warning("No peer prices are available for this context.")
        st.stop()

    recorded_prices = sorted(peers["Actual_Price"].dropna().unique())
    mode = st.radio("Price input", ["Use a recorded price", "Enter a price"], horizontal=True)
    if mode == "Use a recorded price":
        price = st.selectbox("Observed price", recorded_prices, format_func=lambda x: f"${x:,.2f}")
    else:
        price = st.number_input("Price you are seeing", min_value=0.01, value=float(np.median(recorded_prices)), step=0.01, format="%.2f")

    if st.button("Analyze Price", type="primary", use_container_width=True):
        result = analyze_price(price, peers)
        try:
            expected = float(model.predict(make_model_input(context_id))[0])
        except Exception:
            expected = np.nan
        result["ml_expected"] = expected
        result["ml_deviation"] = ((price / expected) - 1) * 100 if np.isfinite(expected) and expected > 0 else np.nan
        st.session_state.last_result = {"context_id": context_id, "label": selected_context_label, "product": product, "retailer": retailer, "store": store, "result": result}
        add_history({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Product": product,
            "Retailer": retailer, "Store": store, "Price": round(float(price), 2), "Signal": result["signal"]
        })

    if "last_result" in st.session_state:
        last = st.session_state.last_result
        r = last["result"]
        st.divider()
        st.markdown("### Price analysis")
        a, b = st.columns([1.05, 1.95])
        with a:
            st.markdown(f'<div class="card"><div class="label">Price signal</div><div style="margin:.5rem 0 .8rem">{signal_pill(r["signal"])}</div><div class="result-price">${r["price"]:,.2f}</div><div class="muted">Selected price</div></div>', unsafe_allow_html=True)
        with b:
            x, y, z = st.columns(3)
            x.metric("Typical peer price", f"${r['median']:,.2f}", delta=f"{r['deviation']:+.2f}%")
            y.metric("Difference vs typical", f"${r['price'] - r['median']:+,.2f}")
            z.metric("Observed range", f"${r['min']:,.2f}–${r['max']:,.2f}")
            st.progress(int(round(r["evidence"])), text=f"Evidence score: {r['evidence']:.0f}/100")

        l, rr = st.columns([1.2, .8])
        with l:
            st.markdown("### Observed peer prices")
            st.plotly_chart(peer_chart(peers, r["price"]), use_container_width=True)
        with rr:
            st.markdown("### Evidence")
            st.metric("Comparison groups", r["count"])
            st.metric("Unique observed prices", r["unique"])
            st.markdown(f'<div class="card"><div class="label">Evidence level</div><div class="big">{r["level"]}</div><div class="muted">Price position: {r["position"]}</div></div>', unsafe_allow_html=True)
            if np.isfinite(r["ml_expected"]):
                st.markdown(f'<div class="card"><div class="label">ML expected price</div><div class="big">${r["ml_expected"]:,.2f}</div><div class="muted">Model deviation: {r["ml_deviation"]:+.2f}%</div></div>', unsafe_allow_html=True)

        st.markdown("### Why did I get this signal?")
        q1, q2, q3 = st.columns(3)
        q1.metric("Above typical peer", f"{r['deviation']:+.2f}%")
        q2.metric("Best observed price", f"${r['min']:,.2f}", delta=f"${r['price'] - r['min']:+,.2f}")
        q3.metric("Context spread", f"{r['spread']:.2f}%")
        st.markdown('<div class="notice">PriceLens identifies unusual pricing patterns from the available observations. A high signal does not by itself establish personalized pricing, discrimination, or intent.</div>', unsafe_allow_html=True)

        st.markdown("### Price scenario")
        lo = max(.01, round(min(r["min"] * .8, r["price"] * .8), 2))
        hi = round(max(r["max"] * 1.25, r["price"] * 1.25, lo + 1), 2)
        scenario_price = st.slider("Move the price to see how the signal changes", lo, hi, float(round(r["price"], 2)), 0.01)
        sr = analyze_price(scenario_price, peers)
        s1, s2, s3 = st.columns(3)
        s1.metric("Scenario price", f"${scenario_price:,.2f}")
        s2.metric("Above typical", f"{sr['deviation']:+.2f}%")
        s3.metric("Scenario signal", sr["signal"])

        download = pd.DataFrame([{
            "Product": last["product"], "Retailer": last["retailer"], "Store": last["store"],
            "Context_ID": last["context_id"], "Selected_Price": r["price"],
            "Typical_Peer_Price": r["median"], "Lowest_Peer_Price": r["min"], "Highest_Peer_Price": r["max"],
            "Price_Above_Typical_Pct": r["deviation"], "Observed_Context_Spread_Pct": r["spread"],
            "Comparison_Groups": r["count"], "Unique_Observed_Prices": r["unique"],
            "Evidence_Score": r["evidence"], "Evidence_Level": r["level"], "Final_Signal": r["signal"],
            "ML_Expected_Price": r["ml_expected"], "ML_Deviation_Pct": r["ml_deviation"],
        }])
        st.download_button("Download price check", download.to_csv(index=False).encode(), "pricelens_price_check.csv", "text/csv")

# ============================================================
# COMPARE
# ============================================================
elif page == "Compare":
    st.title("Compare")
    st.caption("Compare up to three pricing contexts side by side.")
    tmp = context_df.copy()
    tmp["Context_Label"] = tmp.apply(context_label, axis=1)
    cmap = dict(zip(tmp["Context_Label"], tmp["Context_ID"]))
    selected = st.multiselect("Select contexts", list(cmap), max_selections=3)
    if selected:
        ids = [cmap[x] for x in selected]
        view = tmp[tmp["Context_ID"].isin(ids)].copy()
        table = view[["Product_Name", "Retailer", "Store", "Group_Count", "Peer_Unique_Prices", "Lowest_Group_Price", "Peer_Median_Price", "Highest_Group_Price", "Peer_Max_Min_Spread_Pct"]].copy()
        table.columns = ["Product", "Retailer", "Store", "Groups", "Unique prices", "Lowest", "Typical", "Highest", "Spread %"]
        st.dataframe(table.round(2), use_container_width=True, hide_index=True)
        plot = table.melt(id_vars=["Product", "Retailer"], value_vars=["Lowest", "Typical", "Highest"], var_name="Measure", value_name="Price")
        plot["Label"] = plot["Product"].str[:28] + " — " + plot["Retailer"]
        fig = px.bar(plot, x="Label", y="Price", color="Measure", barmode="group", title="Observed price ranges")
        fig.update_layout(template="plotly_white", height=470, xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="card"><div class="label">Compare products or contexts</div><h3>Select up to three contexts above.</h3><div class="muted">Use this view to compare observed ranges and spreads without assigning a fairness judgment.</div></div>', unsafe_allow_html=True)

# ============================================================
# EXPLORE
# ============================================================
elif page == "Explore":
    st.title("Explore the Data")
    st.caption("Investigate the pricing patterns behind PriceLens signals.")
    c1, c2, c3 = st.columns(3)
    retailer = c1.selectbox("Retailer", ["All"] + sorted(group_df["Retailer"].dropna().astype(str).unique()))
    stores = sorted((group_df[group_df["Retailer"].astype(str) == retailer]["Store"] if retailer != "All" else group_df["Store"]).dropna().astype(str).unique())
    store = c2.selectbox("Store", ["All"] + stores)
    signal = c3.selectbox("Signal", ["All", "Low signal", "Moderate signal", "High signal", "Very high signal"])
    e = group_df.copy()
    if retailer != "All": e = e[e["Retailer"].astype(str) == retailer]
    if store != "All": e = e[e["Store"].astype(str) == store]
    if signal != "All": e = e[e["Final_Signal"] == signal]
    if e.empty:
        st.warning("No observations match the selected filters.")
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("Pricing groups", int(e[["Context_ID", "Group"]].drop_duplicates().shape[0]))
    b.metric("Contexts", int(e["Context_ID"].nunique()))
    c.metric("Mean peer deviation", f"{e['Peer_Deviation_Pct'].mean():.2f}%")
    d.metric("Maximum context spread", f"{e['Peer_Max_Min_Spread_Pct'].max():.2f}%")

    l, r = st.columns(2)
    with l:
        sc = e["Final_Signal"].value_counts().reindex(["Low signal", "Moderate signal", "High signal", "Very high signal"], fill_value=0).reset_index()
        sc.columns = ["Signal", "Count"]
        fig = px.bar(sc, x="Signal", y="Count", title="Price signal distribution")
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        spread = e.loc[e["Peer_Max_Min_Spread_Pct"] > 0, "Peer_Max_Min_Spread_Pct"].dropna()
        fig = px.histogram(spread, x=spread, nbins=12, title="Distribution of observed price spreads")
        fig.update_layout(template="plotly_white", height=400, xaxis_title="Maximum–minimum spread (%)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    l, r = st.columns(2)
    with l:
        rs = e.groupby("Retailer")["Peer_Max_Min_Spread_Pct"].mean().reset_index(name="Mean spread")
        fig = px.bar(rs, x="Retailer", y="Mean spread", title="Mean observed spread by retailer")
        fig.update_layout(template="plotly_white", height=420, yaxis_title="Mean spread (%)")
        st.plotly_chart(fig, use_container_width=True)
    with r:
        top = e.sort_values("Peer_Deviation_Pct", ascending=False).head(15).copy()
        top["Product"] = top["Product_Name"].str[:34]
        fig = px.bar(top.sort_values("Peer_Deviation_Pct"), x="Peer_Deviation_Pct", y="Product", orientation="h", title="Top observed price signals")
        fig.update_layout(template="plotly_white", height=420, xaxis_title="Price above peer median (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    plot = e.dropna(subset=["Actual_Price", "OOF_Predicted_Price"])
    fig = px.scatter(plot, x="OOF_Predicted_Price", y="Actual_Price", hover_data=["Product_Name", "Retailer", "Store", "Final_Signal"], title="Observed price vs OOF ML expected price")
    lo = min(plot["OOF_Predicted_Price"].min(), plot["Actual_Price"].min())
    hi = max(plot["OOF_Predicted_Price"].max(), plot["Actual_Price"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash"), name="Perfect prediction"))
    fig.update_layout(template="plotly_white", height=500, xaxis_title="OOF expected price ($)", yaxis_title="Observed price ($)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# INSIGHTS
# ============================================================
elif page == "Insights":
    st.title("Insights")
    st.caption("A high-level view of what the analyzed dataset shows.")
    total = int(context_df["Context_ID"].nunique())
    varying = int((context_df["Peer_Unique_Prices"] > 1).sum())
    rate = varying / total * 100 if total else 0
    largest = float(context_df["Peer_Max_Min_Spread_Pct"].max())

    a, b, c, d = st.columns(4)
    a.metric("Pricing contexts", total)
    b.metric("Contexts with variation", varying)
    c.metric("Variation rate", f"{rate:.2f}%")
    d.metric("Largest observed spread", f"{largest:.2f}%")

    st.markdown("### Key findings")
    st.markdown(
        f'<div class="card"><div class="label">Observed variation</div><h3>69 of {total} pricing contexts contained more than one observed price.</h3><div class="muted">The maximum observed context spread was {largest:.2f}%. These are descriptive findings from this dataset, not a conclusion about intent or legality.</div></div>',
        unsafe_allow_html=True,
    )

    summary = group_df.groupby("Retailer").agg(
        Pricing_Groups=("Group", "size"), Unique_Contexts=("Context_ID", "nunique"),
        Mean_Peer_Deviation=("Peer_Deviation_Pct", "mean"),
        Median_Peer_Deviation=("Peer_Deviation_Pct", "median"),
        Mean_Context_Spread=("Peer_Max_Min_Spread_Pct", "mean"),
        Groups_Above_10Pct=("Peer_Deviation_Pct", lambda x: int((x >= 10).sum())),
    ).reset_index()
    st.markdown("### Retailer view")
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
    fig = px.bar(summary, x="Retailer", y="Mean_Context_Spread", title="Mean observed context spread by retailer")
    fig.update_layout(template="plotly_white", height=430, yaxis_title="Mean spread (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="notice">Retailer comparisons describe this dataset only. PriceLens does not rank retailers or infer that a higher observed variation proves unfairness.</div>', unsafe_allow_html=True)

# ============================================================
# HISTORY
# ============================================================
elif page == "History":
    st.title("History")
    st.caption("Recent checks are stored only for this browser session.")
    history = st.session_state.get("history", [])
    if not history:
        st.markdown('<div class="card"><div class="label">No recent checks</div><h3>Your analyzed prices will appear here.</h3><div class="muted">Start from Check Price to create your first analysis.</div></div>', unsafe_allow_html=True)
    else:
        h = pd.DataFrame(history)
        st.dataframe(h, use_container_width=True, hide_index=True)
        st.download_button("Download history", h.to_csv(index=False).encode(), "pricelens_history.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ============================================================
# HOW IT WORKS
# ============================================================
elif page == "How It Works":
    st.title("How It Works")
    st.caption("The methodology behind the PriceLens prototype.")
    steps = [
        ("01", "Observe", "Use observed grocery pricing data organized into comparable product, retailer, store, date, and time contexts."),
        ("02", "Compare", "Measure where a selected price sits relative to peer prices observed in the same context."),
        ("03", "Predict", "Use a CatBoost regression model to estimate an expected price from observable product and context variables."),
        ("04", "Signal", "Present an interpretable signal using peer deviation, comparison coverage, and the ML estimate."),
    ]
    for n, title, text in steps:
        st.markdown(f'<div class="card"><div class="label">{n} · {title}</div><div style="margin-top:.5rem;line-height:1.6">{text}</div></div>', unsafe_allow_html=True)

    st.markdown("### Model validation")
    a, b, c, d = st.columns(4)
    a.metric("OOF MAE", "$0.71")
    b.metric("OOF RMSE", "$1.18")
    c.metric("OOF R²", "0.90")
    d.metric("Residual ROC-AUC", "0.61")
    st.markdown('<div class="notice">These are grouped out-of-fold validation results. The model is useful for expected-price estimation, but ML residuals alone were not reliable enough to serve as a direct detector of personalized pricing. The peer comparison therefore remains the primary evidence layer.</div>', unsafe_allow_html=True)

    st.markdown("### Dataset and limitations")
    l, r = st.columns(2)
    with l:
        st.markdown('<div class="card"><div class="label">Dataset</div><div class="muted">3,294 observed prices across 96 pricing contexts and 478 experimental pricing groups.</div></div>', unsafe_allow_html=True)
    with r:
        st.markdown('<div class="card"><div class="label">Limitations</div><div class="muted">PriceLens cannot establish why a price differed, whether a consumer was intentionally targeted, or whether a pricing practice is unlawful.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">PriceLens · ML-assisted price anomaly exploration · Prototype</div>', unsafe_allow_html=True)
