from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor


# ============================================================
# PriceLens — ML-assisted price anomaly exploration
# Root-file deployment version
# ============================================================

st.set_page_config(
    page_title="PriceLens",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# All files are expected directly in the GitHub repository root.
MODEL_PATH = BASE_DIR / "price_expectation_model.cbm"
METADATA_PATH = BASE_DIR / "model_metadata.json"

GROUP_PATH = BASE_DIR / "Price_Fairness_Lens_group_scores_final.csv"
CONTEXT_PATH = BASE_DIR / "Price_Fairness_Lens_context_scores.csv"
CONSUMER_PATH = BASE_DIR / "Price_Fairness_Lens_consumer_view.csv"


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "Retailer",
    "Store",
    "Date",
    "Hour",
    "Product_Number",
    "Product_Name",
    "Original_Price",
    "Promotion",
]

CAT_FEATURES = [
    "Retailer",
    "Store",
    "Date",
    "Product_Number",
    "Product_Name",
    "Promotion",
]


# ============================================================
# VISUAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
        '&family=Manrope:wght@600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #f7f8fa;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: Manrope, Inter, sans-serif !important;
        letter-spacing: -.03em;
    }

    .brand {
        font-family: Manrope, sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.05em;
        color: #15171b;
    }

    .tagline {
        color: #747a84;
        font-size: .82rem;
        margin-top: -.15rem;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff, #eef1f4);
        border: 1px solid #e2e5e9;
        border-radius: 24px;
        padding: 3rem 2.2rem;
        margin: 1rem 0 1.5rem;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .72rem;
        font-weight: 700;
        color: #6e7580;
    }

    .hero-title {
        font-family: Manrope, sans-serif;
        font-size: 3.15rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -.055em;
        color: #15171b;
        max-width: 800px;
    }

    .hero-copy {
        color: #626873;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 720px;
        margin-top: .9rem;
    }

    .card {
        background: #fff;
        border: 1px solid #e2e5e9;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 3px 18px rgba(20,24,32,.035);
        margin-bottom: 1rem;
    }

    .label {
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: .7rem;
        color: #777e88;
        font-weight: 700;
    }

    .big {
        font-family: Manrope, sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #181b20;
    }

    .result-price {
        font-family: Manrope, sans-serif;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -.05em;
        color: #181b20;
    }

    .muted {
        color: #737984;
        font-size: .88rem;
        line-height: 1.55;
    }

    .pill {
        display: inline-block;
        padding: .42rem .75rem;
        border-radius: 999px;
        border: 1px solid #dfe3e7;
        background: #f3f5f7;
        color: #30343a;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .03em;
    }

    .notice {
        background: #f2f4f6;
        border-left: 3px solid #616973;
        border-radius: 10px;
        padding: .9rem 1rem;
        color: #505660;
        font-size: .86rem;
        line-height: 1.55;
    }

    .footer {
        color: #8b9199;
        text-align: center;
        font-size: .75rem;
        padding-top: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e2e5e9;
        border-radius: 16px;
        padding: 1rem;
    }

    button {
        border-radius: 12px !important;
    }

    div[data-testid="stRadio"] > div {
        gap: .3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {CONTEXT_PATH.name}"
        )

    if not CONSUMER_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {CONSUMER_PATH.name}"
        )

    context = pd.read_csv(
        CONTEXT_PATH
    )

    consumer = pd.read_csv(
        CONSUMER_PATH
    )

    # --------------------------------------------------------
    # Prefer full group-level file if available.
    # Otherwise build the internal group table from the
    # consumer-view CSV.
    # --------------------------------------------------------

    if GROUP_PATH.exists():

        group = pd.read_csv(
            GROUP_PATH
        )

    else:

        group = consumer.copy()

        group = group.rename(
            columns={
                "Observed_Price":
                    "Actual_Price",

                "Lowest_Peer_Price":
                    "Peer_Group_Min",

                "Typical_Peer_Price":
                    "Peer_Group_Median",

                "Highest_Peer_Price":
                    "Peer_Group_Max",

                "Price_Above_Typical_Pct":
                    "Peer_Deviation_Pct",

                "Context_Price_Spread_Pct":
                    "Peer_Max_Min_Spread_Pct",

                "Comparison_Group_Count":
                    "Peer_Group_Count",
            }
        )

        optional_columns = [
            "Peer_Group_Unique_Prices",
            "Peer_Price_Position",
            "OOF_Predicted_Price",
            "OOF_Model_Deviation_Pct",
            "Group_Respondents",
        ]

        for col in optional_columns:

            if col not in group.columns:

                group[col] = np.nan

    # --------------------------------------------------------
    # Convert group-level numeric fields
    # --------------------------------------------------------

    group_numeric = [

        "Actual_Price",
        "Peer_Group_Count",
        "Peer_Group_Unique_Prices",
        "Peer_Group_Min",
        "Peer_Group_Median",
        "Peer_Group_Max",
        "Peer_Deviation_Pct",
        "Peer_Max_Min_Spread_Pct",
        "Peer_Price_Position",
        "OOF_Predicted_Price",
        "OOF_Model_Deviation_Pct",
        "Evidence_Score",
        "Group_Respondents",

    ]

    for col in group_numeric:

        if col in group.columns:

            group[col] = pd.to_numeric(
                group[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Convert context-level numeric fields
    # --------------------------------------------------------

    context_numeric = [

        "Group_Count",
        "Peer_Unique_Prices",
        "Peer_Min_Price",
        "Peer_Median_Price",
        "Peer_Max_Price",
        "Peer_Max_Min_Spread_Pct",
        "Lowest_Group_Price",
        "Highest_Group_Price",

    ]

    for col in context_numeric:

        if col in context.columns:

            context[col] = pd.to_numeric(
                context[col],
                errors="coerce"
            )

    return group, context, consumer


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing file: {MODEL_PATH.name}"
        )

    model = CatBoostRegressor()

    model.load_model(
        str(MODEL_PATH)
    )

    return model


# ============================================================
# LOAD MODEL METADATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_metadata():

    if not METADATA_PATH.exists():

        return {}

    try:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


# ============================================================
# INITIALIZE DATA + MODEL
# ============================================================

try:

    group_df, context_df, consumer_df = load_data()

    model = load_model()

    model_metadata = load_metadata()

except Exception as exc:

    st.error(
        "PriceLens could not load its model/data files."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# NORMALIZE TEXT COLUMNS
# ============================================================

for df in [
    group_df,
    context_df,
    consumer_df,
]:

    for col in [
        "Retailer",
        "Store",
        "Product_Number",
        "Product_Name",
    ]:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("Unknown")
                .astype(str)
            )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_date(value):

    d = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.notna(d):

        return d.strftime(
            "%Y-%m-%d"
        )

    return "Unknown"


def get_peers(context_id):

    peers = group_df[
        group_df["Context_ID"].astype(str)
        == str(context_id)
    ].copy()

    return peers.sort_values(
        "Actual_Price"
    )


def make_model_input(
    context_row,
    original_price,
    promotion
):

    row = context_row

    x = pd.DataFrame(
        [{
            "Retailer":
                str(
                    row.get(
                        "Retailer",
                        "Unknown"
                    )
                ),

            "Store":
                str(
                    row.get(
                        "Store",
                        "Unknown"
                    )
                ),

            "Date":
                normalize_date(
                    row.get(
                        "Date"
                    )
                ),

            "Hour":
                pd.to_numeric(
                    row.get(
                        "Hour"
                    ),
                    errors="coerce"
                ),

            "Product_Number":
                str(
                    row.get(
                        "Product_Number",
                        "Unknown"
                    )
                ),

            "Product_Name":
                str(
                    row.get(
                        "Product_Name",
                        "Unknown"
                    )
                ),

            "Original_Price":
                pd.to_numeric(
                    original_price,
                    errors="coerce"
                ),

            "Promotion":
                str(
                    promotion
                ),
        }]
    )

    for col in CAT_FEATURES:

        x[col] = (
            x[col]
            .fillna("Unknown")
            .astype(str)
        )

    x["Hour"] = (
        pd.to_numeric(
            x["Hour"],
            errors="coerce"
        )
        .fillna(0)
    )

    x["Original_Price"] = (
        pd.to_numeric(
            x["Original_Price"],
            errors="coerce"
        )
        .fillna(0)
    )

    return x[
        MODEL_FEATURES
    ]


def analyze_price(
    price,
    peers
):

    p = (
        peers["Actual_Price"]
        .dropna()
        .astype(float)
    )

    if p.empty:

        return None

    price = float(
        price
    )

    median = float(
        p.median()
    )

    minimum = float(
        p.min()
    )

    maximum = float(
        p.max()
    )

    count = int(
        p.size
    )

    unique = int(
        p.nunique()
    )

    deviation = (

        (
            (
                price
                /
                median
            )
            - 1
        )
        * 100

        if median > 0

        else 0

    )

    spread = (

        (
            (
                maximum
                /
                minimum
            )
            - 1
        )
        * 100

        if minimum > 0

        else 0

    )

    evidence = float(
        np.clip(
            max(
                deviation,
                0
            )
            /
            25
            *
            100,
            0,
            100
        )
    )

    if count < 3:

        signal = (
            "Insufficient evidence"
        )

    elif deviation <= 5:

        signal = (
            "Low signal"
        )

    elif deviation <= 10:

        signal = (
            "Moderate signal"
        )

    elif deviation <= 20:

        signal = (
            "High signal"
        )

    else:

        signal = (
            "Very high signal"
        )

    if deviation <= 2:

        level = (
            "Within observed peer range"
        )

    elif deviation <= 5:

        level = (
            "Slightly above peers"
        )

    elif deviation <= 10:

        level = (
            "Noticeably above peers"
        )

    elif deviation <= 20:

        level = (
            "Substantially above peers"
        )

    else:

        level = (
            "Strong price anomaly signal"
        )

    if price < minimum:

        position = (
            "Below observed range"
        )

    elif price > maximum:

        position = (
            "Above observed range"
        )

    else:

        position = (
            "Within observed range"
        )

    return {

        "price":
            price,

        "median":
            median,

        "min":
            minimum,

        "max":
            maximum,

        "count":
            count,

        "unique":
            unique,

        "deviation":
            deviation,

        "spread":
            spread,

        "evidence":
            evidence,

        "signal":
            signal,

        "level":
            level,

        "position":
            position,

    }


def context_label(row):

    date = pd.to_datetime(
        row.get(
            "Date"
        ),
        errors="coerce"
    )

    if pd.notna(date):

        date_text = date.strftime(
            "%d %b %Y"
        )

    else:

        date_text = str(
            row.get(
                "Date",
                ""
            )
        )

    try:

        hour = int(
            float(
                row.get(
                    "Hour",
                    0
                )
            )
        )

        time_text = (
            f"{hour:02d}:00"
        )

    except Exception:

        time_text = str(
            row.get(
                "Hour",
                ""
            )
        )

    return (
        f"{row['Product_Name']}  |  "
        f"{row['Retailer']} / "
        f"{row['Store']}  |  "
        f"{date_text} {time_text}  |  "
        f"Context {row['Context_ID']}"
    )


def signal_pill(
    signal
):

    return (
        '<span class="pill">'
        f"{str(signal).upper()}"
        "</span>"
    )


def peer_chart(
    peers,
    selected_price
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=peers[
                "Actual_Price"
            ],

            y=peers[
                "Group"
            ].astype(str),

            mode="markers",

            marker=dict(
                size=13
            ),

            hovertemplate=(
                "$%{x:.2f}"
                "<extra></extra>"
            ),

            name=(
                "Observed peer prices"
            ),

        )
    )

    fig.add_vline(

        x=float(
            selected_price
        ),

        line_dash="dash",

        annotation_text=(
            "Selected price"
        ),
    )

    fig.update_layout(

        template="plotly_white",

        height=350,

        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10
        ),

        xaxis_title="Price ($)",

        yaxis_title=(
            "Pricing group"
        ),

        showlegend=False,

    )

    return fig


def add_history(
    record
):

    history = (
        st.session_state
        .setdefault(
            "history",
            []
        )
    )

    history.insert(
        0,
        record
    )

    st.session_state.history = (
        history[:20]
    )


# ============================================================
# IMPORTANT NAVIGATION FIX
#
# We never directly change the state of the radio widget
# after it has been created. Instead, we store a pending
# navigation request and apply it BEFORE the radio is created
# on the next Streamlit rerun.
# ============================================================

if (
    "pending_nav"
    in st.session_state
):

    st.session_state.nav = (
        st.session_state.pop(
            "pending_nav"
        )
    )


# ============================================================
# TOP NAVIGATION
# ============================================================

nav_col, menu_col = st.columns(
    [1.5, 8.5]
)

with nav_col:

    st.markdown(
        '<div class="brand">'
        'PriceLens'
        '</div>'

        '<div class="tagline">'
        'See beyond the price tag.'
        '</div>',

        unsafe_allow_html=True,
    )


with menu_col:

    pages = [

        "Home",
        "Check Price",
        "Compare",
        "Explore",
        "Insights",
        "History",
        "How It Works",

    ]

    if (
        "nav"
        not in st.session_state
    ):

        st.session_state.nav = (
            "Home"
        )

    st.radio(

        "Navigation",

        pages,

        horizontal=True,

        key="nav",

        label_visibility=(
            "collapsed"
        ),

    )


page = (
    st.session_state.nav
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(

        '<div class="hero">'

        '<div class="eyebrow">'
        'Price intelligence for consumers'
        '</div>'

        '<div class="hero-title">'
        'Is the price you are seeing unusual?'
        '</div>'

        '<div class="hero-copy">'
        'PriceLens compares a selected grocery price '
        'with comparable observed prices and uses a '
        'machine-learning model to estimate an expected price.'
        '</div>'

        '</div>',

        unsafe_allow_html=True,
    )


    if st.button(
        "Check a Price",
        type="primary"
    ):

        st.session_state.pending_nav = (
            "Check Price"
        )

        st.rerun()


    total_contexts = int(
        context_df[
            "Context_ID"
        ].nunique()
    )


    if (
        "Peer_Unique_Prices"
        in context_df.columns
    ):

        varying = int(

            (
                context_df[
                    "Peer_Unique_Prices"
                ]
                > 1
            )
            .sum()

        )

    else:

        varying = int(

            group_df
            .groupby(
                "Context_ID"
            )[
                "Actual_Price"
            ]
            .nunique()
            .gt(1)
            .sum()

        )


    groups = int(

        group_df[
            [
                "Context_ID",
                "Group"
            ]
        ]

        .drop_duplicates()
        .shape[0]

    )


    spread = float(
        context_df[
            "Peer_Max_Min_Spread_Pct"
        ].max()
    )


    st.write("")


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Pricing contexts",
        total_contexts
    )


    c2.metric(
        "Pricing groups",
        groups
    )


    c3.metric(
        "Contexts with variation",
        varying
    )


    c4.metric(
        "Largest observed spread",
        f"{spread:.2f}%"
    )


    left, right = st.columns(
        2
    )


    with left:

        st.markdown(

            '<div class="card">'

            '<div class="label">'
            'What PriceLens does'
            '</div>'

            '<h3>'
            'Compare, predict, explain.'
            '</h3>'

            '<div class="muted">'
            'Direct peer-price comparison is the primary '
            'evidence layer. The CatBoost model provides '
            'a supporting expected-price estimate.'
            '</div>'

            '</div>',

            unsafe_allow_html=True,
        )


    with right:

        st.markdown(

            '<div class="card">'

            '<div class="label">'
            'Important distinction'
            '</div>'

            '<h3>'
            'A signal is not a verdict.'
            '</h3>'

            '<div class="muted">'
            'An unusual observed price can have multiple '
            'explanations. PriceLens does not establish '
            'personalized pricing, discrimination, or intent.'
            '</div>'

            '</div>',

            unsafe_allow_html=True,
        )


# ============================================================
# CHECK PRICE
# ============================================================

elif page == "Check Price":

    st.title(
        "Check a Price"
    )

    st.caption(
        "Choose a comparable context, enter the price you see, "
        "and review the evidence behind the signal."
    )


    products = sorted(

        context_df[
            "Product_Name"
        ]
        .dropna()
        .astype(str)
        .unique()

    )


    product = st.selectbox(
        "Product",
        products
    )


    f = context_df[

        context_df[
            "Product_Name"
        ].astype(str)
        == product

    ].copy()


    retailers = sorted(

        f[
            "Retailer"
        ]
        .dropna()
        .astype(str)
        .unique()

    )


    retailer = st.selectbox(
        "Retailer",
        retailers
    )


    f = f[

        f[
            "Retailer"
        ].astype(str)
        == retailer

    ]


    stores = sorted(

        f[
            "Store"
        ]
        .dropna()
        .astype(str)
        .unique()

    )


    store = st.selectbox(
        "Store",
        stores
    )


    f = f[

        f[
            "Store"
        ].astype(str)
        == store

    ].copy()


    f["Context_Label"] = (
        f.apply(
            context_label,
            axis=1
        )
    )


    cmap = dict(

        zip(

            f[
                "Context_Label"
            ],

            f[
                "Context_ID"
            ]

        )

    )


    selected_context_label = st.selectbox(

        "Pricing context",

        list(cmap)

    )


    context_id = cmap[
        selected_context_label
    ]


    peers = get_peers(
        context_id
    )


    if peers.empty:

        st.warning(
            "No peer prices are available for this context."
        )

        st.stop()


    context_row = f[

        f[
            "Context_ID"
        ]
        == context_id

    ].iloc[0]


    recorded_prices = sorted(

        peers[
            "Actual_Price"
        ]
        .dropna()
        .unique()

    )


    mode = st.radio(

        "Price input",

        [
            "Use a recorded price",
            "Enter a price"
        ],

        horizontal=True,

    )


    if (
        mode
        ==
        "Use a recorded price"
    ):

        price = st.selectbox(

            "Observed price",

            recorded_prices,

            format_func=lambda x:
                f"${x:,.2f}"

        )

    else:

        default_price = float(

            np.median(
                recorded_prices
            )

        )

        price = st.number_input(

            "Price you are seeing",

            min_value=0.01,

            value=default_price,

            step=0.01,

            format="%.2f",

        )


    st.markdown(
        "### ML model inputs"
    )


    st.caption(

        "The saved CatBoost model also uses list/original "
        "price and promotion status. These controls keep "
        "the deployed model fully functional without requiring "
        "the original modelling CSV."

    )


    default_original = float(

        context_row.get(

            "Peer_Median_Price",

            np.median(
                recorded_prices
            )

        )

    )


    m1, m2 = st.columns(
        2
    )


    with m1:

        original_price = st.number_input(

            "List / original price",

            min_value=0.01,

            value=max(
                0.01,
                default_original
            ),

            step=0.01,

            format="%.2f",

        )


    with m2:

        promotion = st.selectbox(

            "Promotion status",

            [
                "Unknown",
                "No promotion",
                "Promotion"
            ]

        )


    if st.button(

        "Analyze Price",

        type="primary",

        use_container_width=True,

    ):

        result = analyze_price(

            price,
            peers

        )


        try:

            x_model = make_model_input(

                context_row,

                original_price,

                promotion

            )


            expected = float(

                model.predict(
                    x_model
                )[0]

            )

        except Exception as exc:

            expected = np.nan

            st.session_state.model_error = (
                str(exc)
            )


        result[
            "ml_expected"
        ] = expected


        result[
            "ml_deviation"
        ] = (

            (

                (

                    float(price)
                    /
                    expected

                )

                - 1

            )
            * 100

            if (

                np.isfinite(
                    expected
                )

                and

                expected > 0

            )

            else np.nan

        )


        if (
            "OOF_Predicted_Price"
            in peers.columns
        ):

            result[
                "validated_oof_expected"
            ] = float(

                peers[
                    "OOF_Predicted_Price"
                ].median()

            )

        else:

            result[
                "validated_oof_expected"
            ] = np.nan


        st.session_state.last_result = {

            "context_id":
                context_id,

            "label":
                selected_context_label,

            "product":
                product,

            "retailer":
                retailer,

            "store":
                store,

            "result":
                result,

            "original_price":
                float(
                    original_price
                ),

            "promotion":
                promotion,

        }


        add_history({

            "Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "Product":
                product,

            "Retailer":
                retailer,

            "Store":
                store,

            "Price":
                round(
                    float(price),
                    2
                ),

            "Signal":
                result[
                    "signal"
                ],

        })


    if (
        "last_result"
        in st.session_state
    ):

        last = (
            st.session_state.last_result
        )

        r = last[
            "result"
        ]


        st.divider()


        st.markdown(
            "### Price analysis"
        )


        a, b = st.columns(
            [
                1.05,
                1.95
            ]
        )


        with a:

            st.markdown(

                '<div class="card">'

                '<div class="label">'
                'Price signal'
                '</div>'

                f'<div style="margin:.5rem 0 .8rem">'
                f'{signal_pill(r["signal"])}'
                '</div>'

                f'<div class="result-price">'
                f'${r["price"]:,.2f}'
                '</div>'

                '<div class="muted">'
                'Selected price'
                '</div>'

                '</div>',

                unsafe_allow_html=True,

            )


        with b:

            x, y, z = st.columns(
                3
            )


            x.metric(

                "Typical peer price",

                f"${r['median']:,.2f}",

                delta=(
                    f"{r['deviation']:+.2f}%"
                ),

            )


            y.metric(

                "Difference vs typical",

                f"${r['price'] - r['median']:+,.2f}",

            )


            z.metric(

                "Observed range",

                f"${r['min']:,.2f}–"
                f"${r['max']:,.2f}",

            )


            st.progress(

                int(
                    round(
                        r["evidence"]
                    )
                ),

                text=(
                    "Prototype evidence score: "
                    f"{r['evidence']:.0f}/100"
                ),

            )


        l, rr = st.columns(
            [
                1.2,
                .8
            ]
        )


        with l:

            st.markdown(
                "### Observed peer prices"
            )


            st.plotly_chart(

                peer_chart(
                    peers,
                    r["price"]
                ),

                use_container_width=True,

            )


        with rr:

            st.markdown(
                "### Evidence"
            )


            st.metric(

                "Comparison groups",

                r["count"]

            )


            st.metric(

                "Unique observed prices",

                r["unique"]

            )


            st.markdown(

                '<div class="card">'

                '<div class="label">'
                'Evidence level'
                '</div>'

                f'<div class="big">'
                f'{r["level"]}'
                '</div>'

                f'<div class="muted">'
                f'Price position: '
                f'{r["position"]}'
                '</div>'

                '</div>',

                unsafe_allow_html=True,

            )


            if np.isfinite(
                r["ml_expected"]
            ):

                st.markdown(

                    '<div class="card">'

                    '<div class="label">'
                    'Live ML expected price'
                    '</div>'

                    f'<div class="big">'
                    f'${r["ml_expected"]:,.2f}'
                    '</div>'

                    f'<div class="muted">'
                    f'Model deviation: '
                    f'{r["ml_deviation"]:+.2f}%'
                    '</div>'

                    '</div>',

                    unsafe_allow_html=True,

                )


            if np.isfinite(

                r.get(

                    "validated_oof_expected",

                    np.nan

                )

            ):

                st.markdown(

                    '<div class="card">'

                    '<div class="label">'
                    'Validated OOF reference'
                    '</div>'

                    f'<div class="big">'
                    f'${r["validated_oof_expected"]:,.2f}'
                    '</div>'

                    '<div class="muted">'
                    'Stored out-of-fold model estimate '
                    'for this context.'
                    '</div>'

                    '</div>',

                    unsafe_allow_html=True,

                )


        st.markdown(
            "### Why did I get this signal?"
        )


        q1, q2, q3 = st.columns(
            3
        )


        q1.metric(

            "Above typical peer",

            f"{r['deviation']:+.2f}%"

        )


        q2.metric(

            "Best observed price",

            f"${r['min']:,.2f}",

            delta=(

                f"${r['price'] - r['min']:+,.2f}"

            ),

        )


        q3.metric(

            "Context spread",

            f"{r['spread']:.2f}%"

        )


        st.markdown(

            '<div class="notice">'
            'PriceLens identifies unusual pricing patterns '
            'from the available observations. A high signal '
            'does not by itself establish personalized pricing, '
            'discrimination, or intent.'
            '</div>',

            unsafe_allow_html=True,

        )


        st.markdown(
            "### Price scenario"
        )


        lo = max(

            0.01,

            round(

                min(

                    r["min"] * .8,

                    r["price"] * .8

                ),

                2

            )

        )


        hi = round(

            max(

                r["max"] * 1.25,

                r["price"] * 1.25,

                lo + 1

            ),

            2

        )


        scenario_price = st.slider(

            "Move the price to see how the signal changes",

            lo,

            hi,

            float(
                round(
                    r["price"],
                    2
                )
            ),

            0.01,

        )


        sr = analyze_price(

            scenario_price,

            peers

        )


        s1, s2, s3 = st.columns(
            3
        )


        s1.metric(

            "Scenario price",

            f"${scenario_price:,.2f}"

        )


        s2.metric(

            "Above typical",

            f"{sr['deviation']:+.2f}%"

        )


        s3.metric(

            "Scenario signal",

            sr["signal"]

        )


        download = pd.DataFrame([{

            "Product":
                last["product"],

            "Retailer":
                last["retailer"],

            "Store":
                last["store"],

            "Context_ID":
                last["context_id"],

            "Selected_Price":
                r["price"],

            "Typical_Peer_Price":
                r["median"],

            "Lowest_Peer_Price":
                r["min"],

            "Highest_Peer_Price":
                r["max"],

            "Price_Above_Typical_Pct":
                r["deviation"],

            "Observed_Context_Spread_Pct":
                r["spread"],

            "Comparison_Groups":
                r["count"],

            "Unique_Observed_Prices":
                r["unique"],

            "Evidence_Score":
                r["evidence"],

            "Evidence_Level":
                r["level"],

            "Final_Signal":
                r["signal"],

            "ML_Expected_Price":
                r["ml_expected"],

            "ML_Deviation_Pct":
                r["ml_deviation"],

            "Validated_OOF_Expected_Price":
                r.get(
                    "validated_oof_expected",
                    np.nan
                ),

            "List_Original_Price_Input":
                last["original_price"],

            "Promotion_Input":
                last["promotion"],

        }])


        st.download_button(

            "Download price check",

            download
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            ),

            "pricelens_price_check.csv",

            "text/csv",

        )


# ============================================================
# COMPARE
# ============================================================

elif page == "Compare":

    st.title(
        "Compare"
    )

    st.caption(
        "Compare up to three pricing contexts side by side."
    )


    tmp = context_df.copy()


    tmp[
        "Context_Label"
    ] = tmp.apply(

        context_label,

        axis=1

    )


    cmap = dict(

        zip(

            tmp[
                "Context_Label"
            ],

            tmp[
                "Context_ID"
            ]

        )

    )


    selected = st.multiselect(

        "Select contexts",

        list(cmap),

        max_selections=3,

    )


    if selected:

        ids = [
            cmap[x]
            for x in selected
        ]


        view = tmp[
            tmp[
                "Context_ID"
            ].isin(ids)
        ].copy()


        table = view[

            [

                "Product_Name",

                "Retailer",

                "Store",

                "Group_Count",

                "Peer_Unique_Prices",

                "Lowest_Group_Price",

                "Peer_Median_Price",

                "Highest_Group_Price",

                "Peer_Max_Min_Spread_Pct",

            ]

        ].copy()


        table.columns = [

            "Product",

            "Retailer",

            "Store",

            "Groups",

            "Unique prices",

            "Lowest",

            "Typical",

            "Highest",

            "Spread %",

        ]


        st.dataframe(

            table.round(
                2
            ),

            use_container_width=True,

            hide_index=True,

        )


        plot = table.melt(

            id_vars=[

                "Product",
                "Retailer"

            ],

            value_vars=[

                "Lowest",
                "Typical",
                "Highest"

            ],

            var_name="Measure",

            value_name="Price",

        )


        plot["Label"] = (

            plot[
                "Product"
            ].str[:28]

            + " — "

            + plot[
                "Retailer"
            ]

        )


        fig = px.bar(

            plot,

            x="Label",

            y="Price",

            color="Measure",

            barmode="group",

            title="Observed price ranges",

        )


        fig.update_layout(

            template="plotly_white",

            height=470,

            xaxis_title="",

            yaxis_title="Price ($)",

        )


        st.plotly_chart(

            fig,

            use_container_width=True,

        )


    else:

        st.markdown(

            '<div class="card">'

            '<div class="label">'
            'Compare products or contexts'
            '</div>'

            '<h3>'
            'Select up to three contexts above.'
            '</h3>'

            '<div class="muted">'
            'Use this view to compare observed ranges '
            'and spreads without assigning a fairness judgment.'
            '</div>'

            '</div>',

            unsafe_allow_html=True,

        )


# ============================================================
# EXPLORE
# ============================================================

elif page == "Explore":

    st.title(
        "Explore the Data"
    )

    st.caption(
        "Investigate the pricing patterns behind PriceLens signals."
    )


    c1, c2, c3 = st.columns(
        3
    )


    retailer_options = [

        "All"

    ] + sorted(

        group_df[
            "Retailer"
        ]
        .dropna()
        .astype(str)
        .unique()

    )


    retailer = c1.selectbox(

        "Retailer",

        retailer_options

    )


    retailer_subset = (

        group_df[
            group_df[
                "Retailer"
            ]
            == retailer
        ]

        if retailer != "All"

        else group_df

    )


    store_options = [

        "All"

    ] + sorted(

        retailer_subset[
            "Store"
        ]
        .dropna()
        .astype(str)
        .unique()

    )


    store = c2.selectbox(

        "Store",

        store_options

    )


    signal = c3.selectbox(

        "Signal",

        [

            "All",

            "Low signal",

            "Moderate signal",

            "High signal",

            "Very high signal",

            "Insufficient evidence",

        ]

    )


    e = group_df.copy()


    if retailer != "All":

        e = e[
            e[
                "Retailer"
            ]
            == retailer
        ]


    if store != "All":

        e = e[
            e[
                "Store"
            ]
            == store
        ]


    if signal != "All":

        e = e[
            e[
                "Final_Signal"
            ]
            == signal
        ]


    if e.empty:

        st.warning(
            "No observations match the selected filters."
        )

        st.stop()


    a, b, c, d = st.columns(
        4
    )


    a.metric(

        "Pricing groups",

        int(

            e[
                [
                    "Context_ID",
                    "Group"
                ]
            ]

            .drop_duplicates()

            .shape[0]

        ),

    )


    b.metric(

        "Contexts",

        int(
            e[
                "Context_ID"
            ].nunique()
        ),

    )


    c.metric(

        "Mean peer deviation",

        f"{e['Peer_Deviation_Pct'].mean():.2f}%"

    )


    d.metric(

        "Maximum context spread",

        f"{e['Peer_Max_Min_Spread_Pct'].max():.2f}%"

    )


    l, r = st.columns(
        2
    )


    with l:

        sc = (

            e[
                "Final_Signal"
            ]

            .value_counts()

            .reindex(

                [

                    "Low signal",

                    "Moderate signal",

                    "High signal",

                    "Very high signal",

                    "Insufficient evidence",

                ],

                fill_value=0

            )

            .reset_index()

        )


        sc.columns = [

            "Signal",
            "Count"

        ]


        fig = px.bar(

            sc,

            x="Signal",

            y="Count",

            title="Price signal distribution",

        )


        fig.update_layout(

            template="plotly_white",

            height=400,

            showlegend=False,

        )


        st.plotly_chart(

            fig,

            use_container_width=True,

        )


    with r:

        spread = e.loc[

            e[
                "Peer_Max_Min_Spread_Pct"
            ]
            > 0,

            "Peer_Max_Min_Spread_Pct"

        ].dropna()


        fig = px.histogram(

            x=spread,

            nbins=12,

            title="Distribution of observed price spreads",

        )


        fig.update_layout(

            template="plotly_white",

            height=400,

            xaxis_title=(
                "Maximum–minimum spread (%)"
            ),

            yaxis_title="Count",

        )


        st.plotly_chart(

            fig,

            use_container_width=True,

        )


    l, r = st.columns(
        2
    )


    with l:

        rs = (

            e.groupby(
                "Retailer"
            )[
                "Peer_Max_Min_Spread_Pct"
            ]

            .mean()

            .reset_index(
                name="Mean spread"
            )

        )


        fig = px.bar(

            rs,

            x="Retailer",

            y="Mean spread",

            title="Mean observed spread by retailer",

        )


        fig.update_layout(

            template="plotly_white",

            height=420,

            yaxis_title=(
                "Mean spread (%)"
            ),

        )


        st.plotly_chart(

            fig,

            use_container_width=True,

        )


    with r:

        top = (

            e.sort_values(

                "Peer_Deviation_Pct",

                ascending=False

            )

            .head(
                15
            )

            .copy()

        )


        top[
            "Product"
        ] = (

            top[
                "Product_Name"
            ].str[:34]

        )


        fig = px.bar(

            top.sort_values(
                "Peer_Deviation_Pct"
            ),

            x="Peer_Deviation_Pct",

            y="Product",

            orientation="h",

            title="Top observed price signals",

        )


        fig.update_layout(

            template="plotly_white",

            height=420,

            xaxis_title=(
                "Price above peer median (%)"
            ),

            yaxis_title="",

        )


        st.plotly_chart(

            fig,

            use_container_width=True,

        )


    if (

        "OOF_Predicted_Price"
        in e.columns

        and

        e[
            "OOF_Predicted_Price"
        ]
        .notna()
        .any()

    ):

        plot = e.dropna(

            subset=[

                "Actual_Price",

                "OOF_Predicted_Price",

            ]

        )


        fig = px.scatter(

            plot,

            x="OOF_Predicted_Price",

            y="Actual_Price",

            hover_data=[

                "Product_Name",

                "Retailer",

                "Store",

                "Final_Signal",

            ],

            title=(
                "Observed price vs OOF ML expected price"
            ),

        )


        lo = min(

            plot[
                "OOF_Predicted_Price"
            ].min(),

            plot[
                "Actual_Price"
            ].min(),

        )


        hi = max(

            plot[
                "OOF_Predicted_Price"
            ].max(),

            plot[
                "Actual_Price"
            ].max(),

        )


        fig.add_trace(

            go.Scatter(

                x=[
                    lo,
                    hi
                ],

                y=[
                    lo,
                    hi
                ],

                mode="lines",

                line=dict(
                    dash="dash"
                ),

                name="Perfect prediction",

            )

        )


        fig.update_layout(

            template="plotly_white",

            height=500,

            xaxis_title=(
                "OOF expected price ($)"
            ),

            yaxis_title=(
                "Observed price ($)"
            ),

        )


        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    else:

        st.info(

            "The deployed compact data does not include "
            "stored OOF predictions. The live ML expected-price "
            "estimate remains available in Check Price."

        )


    st.markdown(
        "### Highest-signal records"
    )


    display_cols = [

        "Product_Name",

        "Retailer",

        "Store",

        "Actual_Price",

        "Peer_Group_Median",

        "Peer_Deviation_Pct",

        "Evidence_Score",

        "Final_Signal",

    ]


    display_cols = [

        col

        for col in display_cols

        if col in e.columns

    ]


    st.dataframe(

        e.sort_values(

            "Evidence_Score",

            ascending=False,

        )[

            display_cols

        ]

        .head(
            25
        )

        .round(
            2
        ),

        use_container_width=True,

        hide_index=True,

    )


# ============================================================
# INSIGHTS
# ============================================================

elif page == "Insights":

    st.title(
        "Insights"
    )

    st.caption(
        "A high-level view of what the analyzed dataset shows."
    )


    total = int(

        context_df[
            "Context_ID"
        ].nunique()

    )


    varying = int(

        (
            context_df[
                "Peer_Unique_Prices"
            ]
            > 1
        ).sum()

        if
        "Peer_Unique_Prices"
        in context_df.columns

        else

        group_df
        .groupby(
            "Context_ID"
        )[
            "Actual_Price"
        ]
        .nunique()
        .gt(1)
        .sum()

    )


    rate = (

        varying
        /
        total
        *
        100

        if total

        else 0

    )


    largest = float(

        context_df[
            "Peer_Max_Min_Spread_Pct"
        ].max()

    )


    a, b, c, d = st.columns(
        4
    )


    a.metric(

        "Pricing contexts",

        total

    )


    b.metric(

        "Contexts with variation",

        varying

    )


    c.metric(

        "Variation rate",

        f"{rate:.2f}%"

    )


    d.metric(

        "Largest observed spread",

        f"{largest:.2f}%"

    )


    st.markdown(
        "### Key findings"
    )


    st.markdown(

        f'<div class="card">'

        f'<div class="label">'
        f'Observed variation'
        f'</div>'

        f'<h3>'
        f'{varying} of {total} pricing contexts contained '
        f'more than one observed price.'
        f'</h3>'

        f'<div class="muted">'
        f'The maximum observed context spread was '
        f'{largest:.2f}%. These are descriptive findings '
        f'from this dataset, not a conclusion about intent '
        f'or legality.'
        f'</div>'

        f'</div>',

        unsafe_allow_html=True,

    )


    summary = (

        group_df

        .groupby(
            "Retailer"
        )

        .agg(

            Pricing_Groups=(
                "Group",
                "size"
            ),

            Unique_Contexts=(
                "Context_ID",
                "nunique"
            ),

            Mean_Peer_Deviation=(
                "Peer_Deviation_Pct",
                "mean"
            ),

            Median_Peer_Deviation=(
                "Peer_Deviation_Pct",
                "median"
            ),

            Mean_Context_Spread=(
                "Peer_Max_Min_Spread_Pct",
                "mean"
            ),

            Groups_Above_10Pct=(

                "Peer_Deviation_Pct",

                lambda x:
                    int(
                        (
                            x >= 10
                        ).sum()
                    )

            ),

        )

        .reset_index()

    )


    st.markdown(
        "### Retailer view"
    )


    st.dataframe(

        summary.round(
            2
        ),

        use_container_width=True,

        hide_index=True,

    )


    fig = px.bar(

        summary,

        x="Retailer",

        y="Mean_Context_Spread",

        title=(
            "Mean observed context spread by retailer"
        ),

    )


    fig.update_layout(

        template="plotly_white",

        height=430,

        yaxis_title=(
            "Mean spread (%)"
        ),

    )


    st.plotly_chart(

        fig,

        use_container_width=True,

    )


    st.markdown(

        '<div class="notice">'
        'Retailer comparisons describe this dataset only. '
        'PriceLens does not rank retailers or infer that a '
        'higher observed variation proves unfairness.'
        '</div>',

        unsafe_allow_html=True,

    )


# ============================================================
# HISTORY
# ============================================================

elif page == "History":

    st.title(
        "History"
    )

    st.caption(
        "Recent checks are stored only for this browser session."
    )


    history = st.session_state.get(
        "history",
        []
    )


    if not history:

        st.markdown(

            '<div class="card">'

            '<div class="label">'
            'No recent checks'
            '</div>'

            '<h3>'
            'Your analyzed prices will appear here.'
            '</h3>'

            '<div class="muted">'
            'Start from Check Price to create your first analysis.'
            '</div>'

            '</div>',

            unsafe_allow_html=True,

        )


    else:

        h = pd.DataFrame(
            history
        )


        st.dataframe(

            h,

            use_container_width=True,

            hide_index=True,

        )


        st.download_button(

            "Download history",

            h.to_csv(
                index=False
            ).encode(
                "utf-8"
            ),

            "pricelens_history.csv",

            "text/csv",

        )


        if st.button(
            "Clear history"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "How It Works":

    st.title(
        "How It Works"
    )

    st.caption(
        "The methodology behind the PriceLens prototype."
    )


    steps = [

        (

            "01",

            "Observe",

            "Use observed grocery pricing data organized "
            "into comparable product, retailer, store, "
            "date, and time contexts."

        ),

        (

            "02",

            "Compare",

            "Measure where a selected price sits relative "
            "to peer prices observed in the same context."

        ),

        (

            "03",

            "Predict",

            "Use the saved CatBoost regression model to "
            "estimate an expected price from observable "
            "product and context variables."

        ),

        (

            "04",

            "Signal",

            "Present an interpretable signal using peer "
            "deviation, comparison coverage, and the "
            "ML estimate."

        ),

    ]


    for n, title, text in steps:

        st.markdown(

            f'<div class="card">'

            f'<div class="label">'
            f'{n} · {title}'
            f'</div>'

            f'<div style="margin-top:.5rem;'
            f'line-height:1.6">'
            f'{text}'
            f'</div>'

            f'</div>',

            unsafe_allow_html=True,

        )


    st.markdown(
        "### Model validation"
    )


    a, b, c, d = st.columns(
        4
    )


    a.metric(
        "OOF MAE",
        "$0.71"
    )


    b.metric(
        "OOF RMSE",
        "$1.18"
    )


    c.metric(
        "OOF R²",
        "0.90"
    )


    d.metric(
        "Residual ROC-AUC",
        "0.61"
    )


    st.markdown(

        '<div class="notice">'
        'These are grouped out-of-fold validation results. '
        'The model is useful for expected-price estimation, '
        'but ML residuals alone were not reliable enough to '
        'serve as a direct detector of personalized pricing. '
        'The peer comparison therefore remains the primary '
        'evidence layer.'
        '</div>',

        unsafe_allow_html=True,

    )


    st.markdown(
        "### Deployment"
    )


    model_version = model_metadata.get(

        "model_type",

        "CatBoostRegressor"

    )


    iterations = model_metadata.get(

        "iterations",

        "-"

    )


    depth = model_metadata.get(

        "depth",

        "-"

    )


    d1, d2, d3 = st.columns(
        3
    )


    d1.metric(

        "Model",

        str(
            model_version
        )

    )


    d2.metric(

        "Iterations",

        str(
            iterations
        )

    )


    d3.metric(

        "Depth",

        str(
            depth
        )

    )


    st.markdown(
        "### Dataset and limitations"
    )


    l, r = st.columns(
        2
    )


    with l:

        st.markdown(

            '<div class="card">'

            '<div class="label">'
            'Dataset'
            '</div>'

            '<div class="muted">'
            '3,294 observed prices across 96 pricing '
            'contexts and 478 experimental pricing groups.'
            '</div>'

            '</div>',

            unsafe_allow_html=True,

        )


    with r:

        st.markdown(

            '<div class="card">'

            '<div class="label">'
            'Limitations'
            '</div>'

            '<div class="muted">'
            'PriceLens cannot establish why a price differed, '
            'whether a consumer was intentionally targeted, '
            'or whether a pricing practice is unlawful.'
            '</div>'

            '</div>',

            unsafe_allow_html=True,

        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(

    '<div class="footer">'
    'PriceLens · ML-assisted price anomaly exploration · Prototype'
    '</div>',

    unsafe_allow_html=True,

)
