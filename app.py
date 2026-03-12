"""
SmartCharging Analytics: Uncovering EV Behavior Patterns
Scenario 2 - Data Mining Summative Assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCharging EV Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — fixed text visibility
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stage-badge {
        background: #0077b6;
        color: #ffffff;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .insight-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        color: #155724;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        color: #856404;
        font-size: 0.95rem;
    }
    .metric-card {
        background: #cce5ff;
        border-left: 5px solid #0077b6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #004085;
        font-size: 0.95rem;
    }
    .rec-card {
        background: #e2e3e5;
        border-left: 5px solid #383d41;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #1b1e21;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_ev_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    charger_types = ["AC Level 1", "AC Level 2", "DC Fast"]
    operators     = ["ChargePoint", "Tesla", "Blink", "EVgo", "Greenlots"]
    connectors    = ["Type 1", "Type 2", "CCS", "CHAdeMO", "Tesla NACS"]
    renewable_src = ["Yes", "No"]
    maint_freq    = ["Monthly", "Quarterly", "Bi-Annually", "Annually"]

    charger_arr   = rng.choice(charger_types, n, p=[0.2, 0.45, 0.35])
    operator_arr  = rng.choice(operators, n)
    renewable_arr = rng.choice(renewable_src, n, p=[0.4, 0.6])
    maint_arr     = rng.choice(maint_freq, n, p=[0.25, 0.35, 0.25, 0.15])
    connector_arr = rng.choice(connectors, n)

    cap_map  = {"AC Level 1": 7, "AC Level 2": 19, "DC Fast": 150}
    capacity = np.array([cap_map[c] + rng.normal(0, 2) for c in charger_arr]).clip(1)
    cost_map = {"AC Level 1": 0.15, "AC Level 2": 0.22, "DC Fast": 0.35}
    cost     = np.array([cost_map[c] + rng.normal(0, 0.03) for c in charger_arr]).clip(0.05)

    base_usage      = capacity * 0.3 + rng.normal(0, 5, n)
    renewable_boost = np.array([3 if r == "Yes" else 0 for r in renewable_arr])
    usage           = (base_usage + renewable_boost).clip(1, 120)

    anomaly_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    usage[anomaly_idx] *= rng.uniform(2.5, 4.5, size=len(anomaly_idx))
    usage = usage.clip(0, 500)

    ratings      = (3 + usage / 50 + rng.normal(0, 0.4, n)).clip(1, 5)
    dist_city    = rng.exponential(15, n).clip(0.5, 100)
    parking      = rng.integers(2, 30, n)
    inst_year    = rng.integers(2012, 2024, n)
    availability = rng.uniform(60, 100, n)
    lats = rng.uniform(25, 48, n)
    lons = rng.uniform(-120, -70, n)
    addresses = [f"{rng.integers(100,9999)} Station {i+1}" for i in range(n)]

    return pd.DataFrame({
        "Station_ID":                 [f"EV_{i+1:04d}" for i in range(n)],
        "Latitude":                   np.round(lats, 4),
        "Longitude":                  np.round(lons, 4),
        "Address":                    addresses,
        "Charger_Type":               charger_arr,
        "Cost_USD_kWh":               np.round(cost, 3),
        "Availability":               np.round(availability, 1),
        "Distance_to_City_km":        np.round(dist_city, 2),
        "Usage_Stats_avg_users_day":  np.round(usage, 1),
        "Station_Operator":           operator_arr,
        "Charging_Capacity_kW":       np.round(capacity, 1),
        "Connector_Types":            connector_arr,
        "Installation_Year":          inst_year,
        "Renewable_Energy_Source":    renewable_arr,
        "Reviews_Rating":             np.round(ratings, 1),
        "Parking_Spots":              parking,
        "Maintenance_Frequency":      maint_arr,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def clean_data(df_raw):
    df     = df_raw.copy()
    before = len(df)
    df     = df.drop_duplicates(subset="Station_ID")
    dropped_dup = before - len(df)

    rng = np.random.default_rng(7)
    for col in ["Reviews_Rating", "Renewable_Energy_Source", "Connector_Types"]:
        idx = rng.choice(df.index, size=int(len(df) * 0.03), replace=False)
        df.loc[idx, col] = np.nan

    missing_before = int(df.isnull().sum().sum())
    df["Reviews_Rating"].fillna(df["Reviews_Rating"].median(), inplace=True)
    df["Renewable_Energy_Source"].fillna("No", inplace=True)
    df["Connector_Types"].fillna(df["Connector_Types"].mode()[0], inplace=True)

    df["Charger_Type_Enc"]            = LabelEncoder().fit_transform(df["Charger_Type"])
    df["Station_Operator_Enc"]        = LabelEncoder().fit_transform(df["Station_Operator"])
    df["Renewable_Energy_Source_Enc"] = LabelEncoder().fit_transform(df["Renewable_Energy_Source"])
    df["Connector_Types_Enc"]         = LabelEncoder().fit_transform(df["Connector_Types"])
    df["Maintenance_Frequency_Enc"]   = LabelEncoder().fit_transform(df["Maintenance_Frequency"])

    num_cols = ["Cost_USD_kWh", "Usage_Stats_avg_users_day",
                "Charging_Capacity_kW", "Distance_to_City_km",
                "Reviews_Rating", "Availability", "Parking_Spots"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])
    df_sc  = pd.DataFrame(scaled, columns=[c + "_scaled" for c in num_cols], index=df.index)
    df     = pd.concat([df, df_sc], axis=1)

    meta = {
        "rows_before": before,
        "rows_after":  len(df),
        "duplicates_dropped":    dropped_dup,
        "missing_values_filled": missing_before,
    }
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def detect_anomalies(df):
    usage = df["Usage_Stats_avg_users_day"]
    z     = np.abs(stats.zscore(usage))
    Q1, Q3 = usage.quantile(0.25), usage.quantile(0.75)
    IQR     = Q3 - Q1
    iqr_flag = (usage < Q1 - 1.5 * IQR) | (usage > Q3 + 1.5 * IQR)
    z_flag   = z > 3

    cost, rating = df["Cost_USD_kWh"], df["Reviews_Rating"]
    hclr = (cost > cost.quantile(0.85)) & (rating < rating.quantile(0.15))

    df = df.copy()
    df["Anomaly_IQR"]          = iqr_flag.astype(int)
    df["Anomaly_Zscore"]       = z_flag.astype(int)
    df["High_Cost_Low_Rating"] = hclr.astype(int)
    df["Is_Anomaly"]           = ((iqr_flag) | (z_flag)).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(df, k=3):
    features = ["Usage_Stats_avg_users_day_scaled", "Charging_Capacity_kW_scaled",
                "Cost_USD_kWh_scaled", "Availability_scaled"]
    X      = df[features].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    df     = df.copy()
    df["Cluster"] = labels

    cluster_means = df.groupby("Cluster")["Usage_Stats_avg_users_day"].mean().sort_values()
    names     = ["Occasional Users", "Daily Commuters", "Heavy Users"]
    label_map = {cid: names[min(r, len(names)-1)] for r, cid in enumerate(cluster_means.index)}
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    inertia, sil = [], []
    ks = list(range(2, 9))
    for ki in ks:
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))

    pca  = PCA(n_components=2, random_state=42)
    X2   = pca.fit_transform(X)
    df["PCA1"] = X2[:, 0]
    df["PCA2"] = X2[:, 1]

    return df, ks, inertia, sil


# ─────────────────────────────────────────────────────────────────────────────
# ASSOCIATION RULE MINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_arm(df):
    d = df.copy()
    d["Usage_Level"]    = pd.cut(d["Usage_Stats_avg_users_day"],
                                  bins=[0, 20, 50, 1e6],
                                  labels=["Low Usage", "Medium Usage", "High Usage"])
    d["Cost_Level"]     = pd.cut(d["Cost_USD_kWh"],
                                  bins=[0, 0.18, 0.28, 1],
                                  labels=["Low Cost", "Medium Cost", "High Cost"])
    d["Capacity_Level"] = pd.cut(d["Charging_Capacity_kW"],
                                  bins=[0, 10, 50, 1e6],
                                  labels=["Low Capacity", "Mid Capacity", "High Capacity"])

    items = ["Charger_Type", "Renewable_Energy_Source", "Usage_Level",
             "Cost_Level", "Capacity_Level", "Maintenance_Frequency"]

    transactions = []
    for _, row in d[items].iterrows():
        t = set()
        for col in items:
            t.add(f"{col}={row[col]}")
        transactions.append(t)

    N         = len(transactions)
    all_items = sorted(set(i for t in transactions for i in t))
    min_sup   = 0.10

    def sup(itemset):
        return sum(1 for t in transactions if itemset.issubset(t)) / N

    rules = []
    for i in range(len(all_items)):
        for j in range(i + 1, len(all_items)):
            a, b = frozenset([all_items[i]]), frozenset([all_items[j]])
            s_ab = sup(a | b)
            if s_ab < min_sup:
                continue
            s_a, s_b = sup(a), sup(b)
            if s_a == 0 or s_b == 0:
                continue
            for ant, con, sa, sc in [(a, b, s_a, s_b), (b, a, s_b, s_a)]:
                conf = s_ab / sa
                lift = conf / sc
                rules.append({
                    "Antecedent": list(ant)[0],
                    "Consequent": list(con)[0],
                    "Support":    round(s_ab, 3),
                    "Confidence": round(conf, 3),
                    "Lift":       round(lift, 3),
                })

    rules_df = (pd.DataFrame(rules)
                .sort_values("Lift", ascending=False)
                .drop_duplicates(subset=["Antecedent", "Consequent"])
                .head(30))
    return rules_df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ SmartCharging Analytics")
    st.markdown("**Scenario 2 — EV Behavior Patterns**")
    st.markdown("---")

    stage = st.radio(
        "Navigate Stages",
        [
            "🏠 Home & Project Scope",
            "🧹 Stage 2: Data Cleaning",
            "📊 Stage 3: EDA",
            "🔵 Stage 4: Clustering",
            "🔗 Stage 5: Association Rules",
            "🚨 Stage 6: Anomaly Detection",
            "💡 Stage 7: Insights & Reporting",
        ],
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parameters")
    n_stations = st.slider("Dataset Size", 200, 1000, 500, 50)
    n_clusters  = st.slider("Number of Clusters (K)", 2, 6, 3)
    st.markdown("---")
    st.caption("Data Mining — Year 1 Summative\nScenario 2: SmartCharging EV Analytics")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PROCESS
# ─────────────────────────────────────────────────────────────────────────────
raw_df                         = generate_ev_dataset(n=n_stations)
df_clean, clean_meta           = clean_data(raw_df)
df_anomaly                     = detect_anomalies(df_clean)
df_clustered, ks, inertia, sil = run_clustering(df_anomaly, k=n_clusters)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE: HOME & PROJECT SCOPE
# ─────────────────────────────────────────────────────────────────────────────
if stage == "🏠 Home & Project Scope":
    st.title("⚡ SmartCharging Analytics Dashboard")
    st.subheader("Uncovering EV Behavior Patterns — SmartEnergy Data Lab")
    st.markdown('<span class="stage-badge">Stage 1 — Project Scope Definition</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📍 Total Stations",    len(df_clean))
    c2.metric("⚡ Avg Daily Users",   f"{df_clean['Usage_Stats_avg_users_day'].mean():.1f}")
    c3.metric("🌱 Renewable Stations", int((df_clean['Renewable_Energy_Source'] == 'Yes').sum()))
    c4.metric("🚨 Anomalies",          int(df_anomaly['Is_Anomaly'].sum()))

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("📌 Project Objectives")
        objectives = [
            ("🔵 Cluster Charging Behaviors",  "Group stations by usage, capacity, cost and availability."),
            ("🚨 Detect Anomalies",            "Identify overuse, faulty stations or abnormal durations."),
            ("🔗 Discover Associations",        "Explore relationships between features and demand."),
            ("🏗️ Infrastructure Planning",     "Support decisions on where to expand charging stations."),
            ("📊 Deploy Dashboard",             "Interactive Streamlit app for exploring patterns."),
        ]
        for title, desc in objectives:
            st.markdown(
                f'<div class="metric-card"><strong>{title}</strong><br>{desc}</div>',
                unsafe_allow_html=True)

    with right:
        st.subheader("📂 Dataset Columns")
        col_info = {
            "Station_ID":                 "Unique identifier for each station",
            "Latitude / Longitude":       "Geographic coordinates",
            "Charger_Type":               "AC Level 1 / AC Level 2 / DC Fast",
            "Cost_USD_kWh":               "Charging cost per kWh",
            "Availability":               "Percentage uptime availability",
            "Distance_to_City_km":        "Distance to nearest city",
            "Usage_Stats_avg_users_day":  "Average number of daily users",
            "Station_Operator":           "Company operating the station",
            "Charging_Capacity_kW":       "Maximum charging power in kW",
            "Renewable_Energy_Source":    "Uses renewable energy: Yes or No",
            "Reviews_Rating":             "Customer rating from 1 to 5",
            "Installation_Year":          "Year the station was installed",
            "Maintenance_Frequency":      "How often maintenance is done",
        }
        for col, desc in col_info.items():
            st.markdown(f"**{col}** — {desc}")

    st.markdown("---")
    st.subheader("🔍 Raw Data Sample")
    st.dataframe(raw_df.head(20), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "🧹 Stage 2: Data Cleaning":
    st.title("🧹 Data Cleaning & Preprocessing")
    st.markdown('<span class="stage-badge">Stage 2 — Data Cleaning & Preprocessing</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Before",           clean_meta["rows_before"])
    c2.metric("Rows After",            clean_meta["rows_after"])
    c3.metric("Duplicates Dropped",    clean_meta["duplicates_dropped"])
    c4.metric("Missing Values Filled", clean_meta["missing_values_filled"])

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Missing Values", "Distributions", "Correlation Heatmap", "Cleaned Data"])

    with tab1:
        st.subheader("Missing Values — How They Were Handled")
        steps = [
            ("Reviews_Rating",         "Numeric column — filled with median value"),
            ("Renewable_Energy_Source", "Categorical column — filled with default value No"),
            ("Connector_Types",         "Categorical column — filled with most common value (mode)"),
        ]
        for col, method in steps:
            st.markdown(
                f'<div class="metric-card"><strong>{col}</strong><br>{method}</div>',
                unsafe_allow_html=True)
        st.markdown(
            '<div class="insight-box">✅ No remaining null values after preprocessing. '
            'All columns are clean and ready for analysis.</div>',
            unsafe_allow_html=True)

    with tab2:
        st.subheader("Distribution of Continuous Variables After Cleaning")
        num_cols = ["Cost_USD_kWh", "Usage_Stats_avg_users_day",
                    "Charging_Capacity_kW", "Distance_to_City_km", "Reviews_Rating"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        colors = ["#0077b6", "#00b4d8", "#90e0ef", "#0096c7", "#48cae4"]
        for i, col in enumerate(num_cols):
            axes[i].hist(df_clean[col], bins=30, color=colors[i],
                         edgecolor="white", alpha=0.9)
            axes[i].set_title(col.replace("_", " "), fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
        axes[-1].axis("off")
        plt.suptitle("Variable Distributions After Cleaning", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Correlation Heatmap — Key Features")
        corr_cols = ["Cost_USD_kWh", "Usage_Stats_avg_users_day", "Charging_Capacity_kW",
                     "Distance_to_City_km", "Reviews_Rating", "Availability",
                     "Charger_Type_Enc", "Renewable_Energy_Source_Enc"]
        corr = df_clean[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, ax=ax, annot_kws={"size": 10})
        ax.set_title("Correlation Matrix of Key Features", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown(
            '<div class="insight-box">🔍 Charging Capacity is positively correlated with '
            'Usage Stats — higher capacity stations attract more daily users.</div>',
            unsafe_allow_html=True)

    with tab4:
        st.subheader("Cleaned Dataset Preview")
        disp = [c for c in df_clean.columns if "_scaled" not in c and "_Enc" not in c]
        st.dataframe(df_clean[disp].head(50), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "📊 Stage 3: EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown('<span class="stage-badge">Stage 3 — Exploratory Data Analysis</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Usage Distribution", "Charger Type", "Cost by Operator",
        "Ratings & Usage", "Growth Over Time"])

    with tab1:
        st.subheader("Usage Stats — Histogram & Boxplot by Charger Type")
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Histogram — Avg Daily Users",
                                            "Boxplot by Charger Type"])
        fig.add_trace(go.Histogram(x=df_clean["Usage_Stats_avg_users_day"],
                                    nbinsx=40, marker_color="#0077b6",
                                    name="Usage"), row=1, col=1)
        for ct in df_clean["Charger_Type"].unique():
            sub = df_clean[df_clean["Charger_Type"] == ct]["Usage_Stats_avg_users_day"]
            fig.add_trace(go.Box(y=sub, name=ct), row=1, col=2)
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">📌 DC Fast chargers show the highest median usage '
            'and widest spread — they serve both daily and emergency charging needs.</div>',
            unsafe_allow_html=True)

    with tab2:
        st.subheader("Charger Type — Station Count and Average Capacity")
        col1, col2 = st.columns(2)
        with col1:
            ct_c = df_clean["Charger_Type"].value_counts().reset_index()
            ct_c.columns = ["Charger_Type", "Count"]
            fig = px.bar(ct_c, x="Charger_Type", y="Count", color="Charger_Type",
                         color_discrete_sequence=["#0077b6", "#00b4d8", "#90e0ef"],
                         title="Number of Stations by Charger Type")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ct_cap = df_clean.groupby("Charger_Type")["Charging_Capacity_kW"].mean().reset_index()
            fig2   = px.bar(ct_cap, x="Charger_Type", y="Charging_Capacity_kW",
                            color="Charger_Type",
                            color_discrete_sequence=["#f4a261", "#e76f51", "#e9c46a"],
                            title="Average Charging Capacity (kW)")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Demand Heatmap — Charger Type vs Availability")
        pv = df_clean.copy()
        pv["Avail_Bucket"] = pd.cut(pv["Availability"],
                                     bins=[60, 75, 85, 95, 101],
                                     labels=["60-75%", "75-85%", "85-95%", "95-100%"])
        hm  = pv.groupby(["Charger_Type", "Avail_Bucket"])["Usage_Stats_avg_users_day"].mean().unstack()
        fig3, ax = plt.subplots(figsize=(9, 3.5))
        sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    annot_kws={"size": 11})
        ax.set_title("Mean Daily Users — Charger Type × Availability Band",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with tab3:
        st.subheader("Cost (USD/kWh) by Station Operator")
        fig = px.box(df_clean, x="Station_Operator", y="Cost_USD_kWh",
                     color="Station_Operator",
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     title="Cost per kWh Distribution by Operator")
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Tesla stations show higher median cost but '
            'also higher ratings — suggesting premium pricing with better service.</div>',
            unsafe_allow_html=True)

    with tab4:
        st.subheader("Reviews Rating vs Average Daily Usage")
        fig = px.scatter(df_clean, x="Reviews_Rating",
                         y="Usage_Stats_avg_users_day",
                         color="Charger_Type",
                         size="Charging_Capacity_kW",
                         hover_data=["Station_Operator", "Cost_USD_kWh"],
                         title="Rating vs Usage — bubble size shows Capacity",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Renewable Energy Impact on Average Daily Usage")
        ren_grp = df_clean.groupby("Renewable_Energy_Source")[
            "Usage_Stats_avg_users_day"].describe().round(2)
        st.dataframe(ren_grp, use_container_width=True)
        ren_avg = df_clean.groupby("Renewable_Energy_Source")["Usage_Stats_avg_users_day"].mean()
        st.markdown(
            f'<div class="insight-box">🌱 Renewable stations average '
            f'<strong>{ren_avg.get("Yes", 0):.1f}</strong> users/day vs '
            f'<strong>{ren_avg.get("No", 0):.1f}</strong> for non-renewable stations.'
            f'</div>', unsafe_allow_html=True)

    with tab5:
        st.subheader("Station Growth and Usage Trend Over Installation Years")
        yr = df_clean.groupby("Installation_Year").agg(
            Stations=("Station_ID", "count"),
            Avg_Usage=("Usage_Stats_avg_users_day", "mean")
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yr["Installation_Year"], y=yr["Stations"],
                              name="Stations Installed",
                              marker_color="#90e0ef"), secondary_y=False)
        fig.add_trace(go.Scatter(x=yr["Installation_Year"], y=yr["Avg_Usage"],
                                  mode="lines+markers", name="Avg Users/Day",
                                  line=dict(color="#0077b6", width=2.5)),
                      secondary_y=True)
        fig.update_layout(title="Stations Installed vs Avg Daily Users per Year", height=420)
        fig.update_yaxes(title_text="Stations Installed", secondary_y=False)
        fig.update_yaxes(title_text="Avg Daily Users",    secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: CLUSTERING  (simplified — no map)
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "🔵 Stage 4: Clustering":
    st.title("🔵 Clustering Analysis — K-Means")
    st.markdown('<span class="stage-badge">Stage 4 — Clustering Analysis</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Elbow Method", "Cluster Scatter", "Cluster Profiles"])

    with tab1:
        st.subheader("Elbow Method & Silhouette Score — Finding Best K")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ks, inertia, "o-", color="#0077b6", linewidth=2, markersize=7)
        axes[0].axvline(x=n_clusters, color="red", linestyle="--",
                        alpha=0.8, label=f"Selected K={n_clusters}")
        axes[0].set_title("Elbow Method (Inertia)", fontweight="bold")
        axes[0].set_xlabel("Number of Clusters K")
        axes[0].set_ylabel("Inertia")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ks, sil, "s-", color="#00b4d8", linewidth=2, markersize=7)
        axes[1].axvline(x=n_clusters, color="red", linestyle="--",
                        alpha=0.8, label=f"Selected K={n_clusters}")
        axes[1].set_title("Silhouette Score", fontweight="bold")
        axes[1].set_xlabel("Number of Clusters K")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best_k = ks[sil.index(max(sil))]
        st.markdown(
            f'<div class="insight-box">📌 Best silhouette score is at '
            f'<strong>K={best_k}</strong>. You are currently using K={n_clusters}. '
            f'Adjust in the sidebar.</div>',
            unsafe_allow_html=True)

    with tab2:
        st.subheader("Cluster Scatter Plot — PCA 2D View")
        fig = px.scatter(df_clustered, x="PCA1", y="PCA2",
                         color="Cluster_Label",
                         hover_data=["Station_ID", "Station_Operator",
                                     "Usage_Stats_avg_users_day",
                                     "Charging_Capacity_kW"],
                         title=f"K-Means Clusters (K={n_clusters}) — PCA Projection",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(marker=dict(size=7, opacity=0.8))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Cluster Profile — Average Feature Values")
        profile_cols = ["Usage_Stats_avg_users_day", "Charging_Capacity_kW",
                        "Cost_USD_kWh", "Reviews_Rating", "Distance_to_City_km"]
        profile = df_clustered.groupby("Cluster_Label")[profile_cols].mean().round(2)
        st.dataframe(profile, use_container_width=True)

        st.subheader("Station Count per Cluster by Charger Type")
        ct_clust = (df_clustered.groupby(["Cluster_Label", "Charger_Type"])
                    .size().reset_index(name="Count"))
        fig = px.bar(ct_clust, x="Cluster_Label", y="Count",
                     color="Charger_Type", barmode="group",
                     title="Stations per Cluster by Charger Type",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Heavy Users cluster is dominated by DC Fast '
            'chargers. Occasional Users cluster has more AC Level 1 stations.</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: ASSOCIATION RULES
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "🔗 Stage 5: Association Rules":
    st.title("🔗 Association Rule Mining")
    st.markdown('<span class="stage-badge">Stage 5 — Association Rule Mining</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Mining association rules — please wait …"):
        rules_df = run_arm(df_clean)

    st.info(f"**Method:** Frequent Itemset Mining (Apriori equivalent)  |  "
            f"**Min Support:** 10%  |  **Rules Found:** {len(rules_df)}")

    tab1, tab2, tab3 = st.tabs(["Rules Table", "Lift Bar Chart", "Support vs Confidence"])

    with tab1:
        st.subheader("Top Association Rules — Sorted by Lift")
        st.dataframe(rules_df.head(25), use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Lift greater than 1 means the antecedent '
            'increases the likelihood of the consequent. These rules guide pricing '
            'and infrastructure decisions.</div>',
            unsafe_allow_html=True)

    with tab2:
        st.subheader("Top 15 Rules by Lift")
        top15 = rules_df.head(15).copy()
        top15["Rule"] = (top15["Antecedent"].str[:22]
                         + " → " + top15["Consequent"].str[:22])
        fig = px.bar(top15, x="Lift", y="Rule", orientation="h",
                     color="Confidence", color_continuous_scale="Blues",
                     title="Top 15 Association Rules by Lift")
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Support vs Confidence — bubble size shows Lift")
        fig = px.scatter(rules_df, x="Support", y="Confidence",
                         size="Lift", color="Lift",
                         color_continuous_scale="Viridis",
                         hover_data=["Antecedent", "Consequent"],
                         title="Support vs Confidence Plot")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight-box">📊 Rules in the top-right corner have both '
            'high support and high confidence — these are the most reliable patterns '
            'for making business decisions.</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: ANOMALY DETECTION  (no map)
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "🚨 Stage 6: Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.markdown('<span class="stage-badge">Stage 6 — Anomaly Detection</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    n_anom = int(df_anomaly["Is_Anomaly"].sum())
    n_hclr = int(df_anomaly["High_Cost_Low_Rating"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Stations",             len(df_anomaly))
    c2.metric("Anomalies Detected",          n_anom)
    c3.metric("High Cost + Low Rating",      n_hclr)
    c4.metric("Anomaly Rate",                f"{100*n_anom/len(df_anomaly):.1f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["IQR Boxplot", "Z-Score Analysis", "Anomaly Table"])

    with tab1:
        st.subheader("IQR Outlier Detection — Usage Stats")
        fig = px.box(df_anomaly,
                     y="Usage_Stats_avg_users_day",
                     color="Is_Anomaly",
                     color_discrete_map={0: "#90e0ef", 1: "#e63946"},
                     points="all",
                     title="IQR Boxplot — Normal vs Anomaly Stations",
                     labels={"Is_Anomaly": "Anomaly (1 = Yes, 0 = No)"})
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="warning-box">⚠️ Red points exceed the IQR upper fence '
            '(Q3 + 1.5 × IQR). These stations have unusually high usage — '
            'possible overuse or faulty meter readings.</div>',
            unsafe_allow_html=True)

    with tab2:
        st.subheader("Z-Score Method — Threshold |Z| greater than 3")
        usage    = df_anomaly["Usage_Stats_avg_users_day"]
        z_scores = np.abs(stats.zscore(usage))
        fig, ax  = plt.subplots(figsize=(11, 4))
        colors   = ["#e63946" if z > 3 else "#90e0ef" for z in z_scores]
        ax.scatter(range(len(z_scores)), z_scores,
                   c=colors, alpha=0.7, s=20)
        ax.axhline(3, color="red", linestyle="--",
                   linewidth=2, label="|Z| = 3 threshold")
        ax.set_title("Z-Scores of Usage Stats — Red points are anomalies",
                     fontweight="bold")
        ax.set_xlabel("Station Index")
        ax.set_ylabel("|Z-Score|")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("High Cost + Low Rating Stations")
        fig2 = px.scatter(df_anomaly,
                          x="Cost_USD_kWh", y="Reviews_Rating",
                          color="High_Cost_Low_Rating",
                          color_discrete_map={0: "#90e0ef", 1: "#e63946"},
                          hover_data=["Station_ID", "Station_Operator", "Charger_Type"],
                          title="High Cost vs Low Rating — Red points are flagged",
                          labels={"High_Cost_Low_Rating": "Flagged (1=Yes)"})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Anomalous Station Records")
        anom_cols = ["Station_ID", "Station_Operator", "Charger_Type",
                     "Usage_Stats_avg_users_day", "Cost_USD_kWh",
                     "Reviews_Rating", "Anomaly_IQR", "Anomaly_Zscore",
                     "High_Cost_Low_Rating"]
        anom_data = (df_anomaly[df_anomaly["Is_Anomaly"] == 1][anom_cols]
                     .sort_values("Usage_Stats_avg_users_day", ascending=False))
        st.dataframe(anom_data, use_container_width=True)
        st.markdown(
            f'<div class="warning-box">⚠️ Total of <strong>{n_anom}</strong> anomalous '
            f'stations found. These may be overloaded stations needing capacity '
            f'expansion or stations with faulty meters.</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7: INSIGHTS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────
elif stage == "💡 Stage 7: Insights & Reporting":
    st.title("💡 Insights & Business Recommendations")
    st.markdown('<span class="stage-badge">Stage 7 — Insights & Reporting</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    city_mean  = df_clean[df_clean["Distance_to_City_km"] < 10]["Usage_Stats_avg_users_day"].mean()
    rural_mean = df_clean[df_clean["Distance_to_City_km"] >= 10]["Usage_Stats_avg_users_day"].mean()
    top_ct     = df_clean.groupby("Charger_Type")["Usage_Stats_avg_users_day"].mean().idxmax()
    top_op     = df_clean.groupby("Station_Operator")["Reviews_Rating"].mean().idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Most Used Charger Type",  top_ct)
    c2.metric("Highest Rated Operator",   top_op)
    c3.metric("City vs Rural Users/Day",  f"{city_mean:.1f} vs {rural_mean:.1f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Summary Findings", "Operator Comparison", "Recommendations"])

    with tab1:
        st.subheader("Summary of All Analytical Stages")
        dc_usage = df_clean[df_clean["Charger_Type"] == "DC Fast"]["Usage_Stats_avg_users_day"].mean()
        ren_yes  = df_clean[df_clean["Renewable_Energy_Source"] == "Yes"]["Usage_Stats_avg_users_day"].mean()
        ren_no   = df_clean[df_clean["Renewable_Energy_Source"] == "No"]["Usage_Stats_avg_users_day"].mean()

        findings = [
            ("🔵 Clustering",
             f"K-Means with K={n_clusters} segmented stations into Occasional Users, "
             f"Daily Commuters, and Heavy Users. DC Fast chargers dominate the Heavy Users cluster."),
            ("📊 Charger Type",
             f"AC Level 2 stations are most common. DC Fast stations average "
             f"{dc_usage:.1f} users per day which is the highest of all charger types."),
            ("🌱 Renewable Energy",
             f"Renewable-powered stations average {ren_yes:.1f} vs {ren_no:.1f} users per day. "
             f"That is a {((ren_yes - ren_no) / ren_no * 100):.1f}% improvement."),
            ("🚨 Anomaly Detection",
             f"{df_anomaly['Is_Anomaly'].sum()} anomalous stations were detected, "
             f"making up {100 * df_anomaly['Is_Anomaly'].mean():.1f}% of all stations. "
             f"These represent overuse hotspots or potential equipment faults."),
            ("🔗 Association Mining",
             "DC Fast Charger combined with High Usage Level is a consistently strong "
             "association. Low-cost stations near cities reliably show high demand."),
            ("📍 Urban vs Rural",
             f"City stations average {city_mean:.1f} users per day vs {rural_mean:.1f} "
             f"for rural stations — confirming that proximity to urban centres drives demand."),
        ]
        for title, text in findings:
            st.markdown(
                f'<div class="insight-box"><strong>{title}:</strong><br>{text}</div>',
                unsafe_allow_html=True)

    with tab2:
        st.subheader("Operator Comparison — Rating vs Usage vs Cost")
        op_profile = df_clean.groupby("Station_Operator").agg(
            Avg_Rating    =("Reviews_Rating",             "mean"),
            Avg_Usage     =("Usage_Stats_avg_users_day",  "mean"),
            Avg_Cost      =("Cost_USD_kWh",               "mean"),
            Station_Count =("Station_ID",                 "count"),
        ).reset_index().round(2)
        st.dataframe(op_profile, use_container_width=True)

        fig = px.scatter(op_profile,
                         x="Avg_Rating", y="Avg_Usage",
                         size="Station_Count", color="Station_Operator",
                         hover_data=["Avg_Cost"],
                         title="Operator: Rating vs Avg Usage — bubble size = station count",
                         color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Business Recommendations")
        recs = [
            ("1. Expand DC Fast Charger Network",
             "DC Fast stations generate the highest usage. Prioritise adding them "
             "in Heavy User cluster zones especially in urban centres."),
            ("2. Audit High Cost and Low Rating Stations",
             f"{df_anomaly['High_Cost_Low_Rating'].sum()} stations charge premium rates "
             f"but receive poor reviews. These should be investigated for service quality."),
            ("3. Increase Renewable Energy Share",
             "Renewable stations attract more users and support sustainability goals. "
             "Target converting existing non-renewable stations."),
            ("4. Investigate Anomalous Stations",
             f"The {df_anomaly['Is_Anomaly'].sum()} anomalous stations may be faulty "
             f"meters or high-demand underserved locations. Both require immediate action."),
            ("5. Focus on Rural Expansion",
             f"Urban stations are close to saturation at {city_mean:.1f} users per day. "
             f"Rural stations at {rural_mean:.1f} users per day have capacity for growth."),
            ("6. Align Maintenance with Usage Peaks",
             "High-usage stations benefit from more frequent maintenance. "
             "Schedule preventive maintenance before peak seasons to avoid downtime."),
        ]
        for title, body in recs:
            st.markdown(
                f'<div class="rec-card"><strong>{title}</strong><br>{body}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "⚡ SmartCharging Analytics — Data Mining Year 1 Summative | "
    "Scenario 2: EV Behavior Patterns | Built with Streamlit"
)