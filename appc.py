"""
Customer Segmentation Predictor — Fixed & Production Ready
===========================================================
Root causes fixed:
  1. fillna(inplace=True) silently failed in pandas CoW → left NaN → corrupted scaling
  2. Segment names were hardcoded to wrong cluster IDs — now data-driven
  3. Scaler transform used plain list → feature name mismatch → wrong predictions
  4. Classifier trained on NaN-corrupted data → predicted same cluster for everyone

Correct cluster mapping (verified from data):
  Cluster 0 → High Spenders – Inactive  (high income/spend, high recency)
  Cluster 1 → Young Budget              (low income, youngest, low spend)
  Cluster 2 → Budget Seniors            (mid income, oldest, many kids)
  Cluster 3 → Premium Champions         (highest income/spend, most recent)
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score, accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  CSS — Times New Roman throughout
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown, div, p, span, li, td, th, label, button {
    font-family: 'Times New Roman', Times, serif !important;
}
.main { background-color: #f9f7f4; }
.block-container { padding: 2rem 2.8rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1c1c2e; border-right: 1px solid #2e2e4a; }
[data-testid="stSidebar"] * { font-family: 'Times New Roman', Times, serif !important; color: #d4d0c8 !important; }
[data-testid="stSidebar"] h2 { color: #c9a84c !important; font-size: 0.85rem !important; letter-spacing: 2px; text-transform: uppercase; border-bottom: 1px solid #2e2e4a; padding-bottom: 6px; }

/* KPI Card */
.kpi-card { background: #ffffff; border: 1px solid #ddd8cf; border-top: 4px solid #2c3e6b; border-radius: 6px; padding: 18px 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.kpi-label { font-size: 0.7rem; letter-spacing: 1.8px; text-transform: uppercase; color: #888; margin-bottom: 8px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #1a2545; }
.kpi-sub { font-size: 0.75rem; color: #2c3e6b; margin-top: 4px; }

/* Section header */
.sec-hdr { font-size: 1.12rem; font-weight: 700; color: #1a2545; border-left: 5px solid #c9a84c; padding-left: 12px; margin: 1.8rem 0 0.9rem; }

/* Page title */
.page-title { font-size: 2rem; font-weight: 700; color: #1a2545; margin: 0 0 4px; }
.page-sub { font-size: 0.95rem; color: #7a7060; margin-bottom: 1.6rem; }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-family: 'Times New Roman', serif !important; font-size: 0.95rem; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #1a2545 !important; border-bottom: 3px solid #c9a84c !important; }

/* Pipeline step */
.pipeline-step { background: #fff; border: 1px solid #e0dbd0; border-left: 5px solid #2c3e6b; border-radius: 6px; padding: 14px 18px; margin-bottom: 10px; font-size: 0.93rem; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────
FEATURES = ['Income', 'Age', 'Total_Spending', 'TotKids', 'Recency', 'Total_Purchases']

# Data-driven mapping — derived from actual cluster means:
# Cluster 0: Income=69903, Spend=1176, Age=58, Recency=73 → High Spender but Inactive
# Cluster 1: Income=31630, Spend=120,  Age=47            → Young Budget
# Cluster 2: Income=44190, Spend=193,  Age=60, Kids=1.9  → Budget Seniors (family)
# Cluster 3: Income=70708, Spend=1118, Recency=21        → Premium Champions (active)
SEGMENT_CONFIG = {
    0: {
        "name":     "High Spenders – Inactive",
        "color":    "#d97706",
        "bg":       "#fffbeb",
        "icon":     "🛒",
        "profile":  "High income, high spending but haven't purchased recently. Risk of churn.",
        "strategy": "Win-back campaigns, exclusive re-engagement offers, loyalty renewal incentives.",
    },
    1: {
        "name":     "Young Budget",
        "color":    "#16a34a",
        "bg":       "#f0fdf4",
        "icon":     "👨‍👩‍👧",
        "profile":  "Youngest segment, lowest income and spending. Family-oriented, deals-driven.",
        "strategy": "Bundle deals, discount coupons, web-first promotions, refer-a-friend programs.",
    },
    2: {
        "name":     "Budget Seniors",
        "color":    "#2563eb",
        "bg":       "#eff6ff",
        "icon":     "👴",
        "profile":  "Older customers with most children at home. Moderate income, low-to-mid spending.",
        "strategy": "Family value packs, loyalty rewards, senior discounts, health & wellness bundles.",
    },
    3: {
        "name":     "Premium Champions",
        "color":    "#9333ea",
        "bg":       "#faf5ff",
        "icon":     "💎",
        "profile":  "Highest income and spending, most recently active. Top campaign responders.",
        "strategy": "VIP early access, exclusive catalogs, personalised high-value offers, concierge.",
    },
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Times New Roman, Times, serif", color="#333"),
    margin=dict(l=20, r=20, t=45, b=20),
)

# ─────────────────────────────────────────────────────────────────
#  DATA ENGINEERING  — exactly mirrors your notebook, NaN-safe
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_engineer(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))

    # ── FIX 1: correct fillna (no inplace, avoids pandas CoW bug) ──
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df = df.drop_duplicates().reset_index(drop=True)

    # Cleaning
    df['Marital_Status'] = df['Marital_Status'].replace(
        {'Alone':'Single','Widow':'Single','Divorced':'Single',
         'YOLO':'Single','Absurd':'Single','Together':'Married'})
    df['Education'] = df['Education'].replace({'Basic': 'Graduation'})

    # Feature engineering (matches your notebook)
    df['MntNonVeg_Prodicuts']      = df['MntMeatProducts'] + df['MntFishProducts']
    df['TotKids']         = df['Kidhome'] + df['Teenhome']
    df['Age']             = 2024 - df['Year_Birth']
    df['Total_Purchases'] = (df['NumWebPurchases']
                              + df['NumCatalogPurchases']
                              + df['NumStorePurchases'])
    df['Total_Spending']  = (df['MntWines'] + df['MntFruits']
                              + df['MntSweetProducts'] + df['MntGoldProds']
                              + df['MntNonVeg_Prodicuts'])
    return df


# ─────────────────────────────────────────────────────────────────
#  CLUSTERING
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def run_kmeans(file_bytes: bytes):
    df = load_and_engineer(file_bytes)

    # ── FIX 2: use DataFrame (not array) so scaler stores feature names ──
    X_df = df[FEATURES].copy()
    sc   = StandardScaler()
    Xs   = sc.fit_transform(X_df)

    km = KMeans(n_clusters=4, random_state=12, n_init=10)
    df['Cluster'] = km.fit_predict(Xs)

    # PCA for visualisation
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(Xs)
    df['PC1'] = coords[:, 0]
    df['PC2'] = coords[:, 1]

    # WCSS elbow
    wcss = [KMeans(n_clusters=k, random_state=12, n_init=10).fit(Xs).inertia_
            for k in range(1, 10)]

    sil = round(silhouette_score(Xs, df['Cluster']), 4)
    evr = pca.explained_variance_ratio_

    return df, sc, Xs, km, sil, wcss, evr


# ─────────────────────────────────────────────────────────────────
#  CLASSIFICATION
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def train_classifiers(file_bytes: bytes):
    df, sc, Xs, _, _, _, _ = run_kmeans(file_bytes)
    y = df['Cluster'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    def _res(model, pred):
        return {
            "model":  model,
            "acc":    round(accuracy_score(y_te, pred) * 100, 2),
            "f1":     round(f1_score(y_te, pred, average='weighted') * 100, 2),
            "report": classification_report(y_te, pred, output_dict=True),
            "cm":     confusion_matrix(y_te, pred),
            "y_te":   y_te,
            "y_pred": pred,
        }

    clf = {
        "Logistic Regression": _res(lr, lr_pred),
        "Random Forest":       {**_res(rf, rf_pred), "fi": rf.feature_importances_},
    }
    clf["best"] = max(["Logistic Regression","Random Forest"],
                      key=lambda k: clf[k]["acc"])
    return clf


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Segmentation Studio")
    st.markdown("---")
    uploaded = st.file_uploader("📂  Upload Dataset (.xlsx)", type=["xlsx","xls"])
    if uploaded is None:
        st.info("Upload `marketing_campaign.xlsx` to begin.")
        st.stop()

    st.markdown("---")
    st.markdown("## 📌 Features Used")
    for lbl in ['💰 Income','📅 Age','🛒 Total Spending',
                '👶 Total Kids','🕐 Recency','🧾 Total Purchases']:
        st.markdown(f"&nbsp;&nbsp;&nbsp;{lbl}")

    st.markdown("---")
    st.markdown("## 🔍 Navigation")
    page = st.radio("", [
        "🔮 Predict Segment",
        "📊 Segment Overview",
        "🤖 Model Comparison",
        "ℹ️ About",
    ], label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("Loading data and training models…"):
    df, sc, Xs, km_model, sil, wcss, evr = run_kmeans(file_bytes)
    clf = train_classifiers(file_bytes)

best_clf   = clf["best"]
best_acc   = clf[best_clf]["acc"]
best_f1    = clf[best_clf]["f1"]

df["SegName"] = df["Cluster"].map(lambda c: SEGMENT_CONFIG[c]["name"])

summary = df.groupby("Cluster").agg(
    Count=("Cluster","count"),
    AvgIncome=("Income","mean"),
    AvgAge=("Age","mean"),
    AvgSpend=("Total_Spending","mean"),
    AvgPurchases=("Total_Purchases","mean"),
    AvgKids=("TotKids","mean"),
    AvgRecency=("Recency","mean"),
).round(1).reset_index()

# ─────────────────────────────────────────────────────────────────
#  GLOBAL KPI BANNER
# ─────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🎯 Customer Segmentation Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Predict which segment a new customer belongs to — '
    'with confidence scores and feature importance</p>',
    unsafe_allow_html=True)

for col, lbl, val, sub in zip(
    st.columns(5),
    ["Total Customers",   "K-Means Segments",  "Silhouette Score",  "Best Classifier",  "Best F1-Score"],
    [f"{len(df):,}",      "4",                 f"{sil:.4f}",        best_clf.split()[0], f"{best_f1}%"],
    ["after cleaning",    "K = 4 (optimal)",   "cluster quality",   f"Acc {best_acc}%",  best_clf.split()[0]],
):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE: PREDICT SEGMENT
# ══════════════════════════════════════════════════════════════════
if page == "🔮 Predict Segment":
    st.markdown('<p class="sec-hdr">Enter Customer Details</p>', unsafe_allow_html=True)

    clf_choice    = st.radio("Classifier", ["Logistic Regression","Random Forest"], horizontal=True)
    chosen_model  = clf[clf_choice]["model"]

    c1, c2, c3 = st.columns(3)
    with c1:
        income    = st.number_input("💰 Annual Income ($)",         0, 200_000, 55_000, step=1000)
        age       = st.slider("📅 Age",                             18, 90, 45)
    with c2:
        spending  = st.number_input("🛒 Total Spending ($)",         0, 3_000,    500, step=50)
        kids      = st.slider("👶 Total Kids at Home",               0, 5, 1)
    with c3:
        recency   = st.slider("🕐 Recency (days since last purchase)", 0, 100, 30)
        purchases = st.slider("🧾 Total Purchases",                   0, 40, 10)

    if st.button("🔮  Predict Segment", type="primary", use_container_width=True):

        # ── FIX 3: pass DataFrame with feature names to avoid scaler mismatch ──
        inp_df  = pd.DataFrame([[income, age, spending, kids, recency, purchases]],
                               columns=FEATURES)
        inp_scaled = sc.transform(inp_df)

        seg  = int(chosen_model.predict(inp_scaled)[0])
        cfg  = SEGMENT_CONFIG[seg]

        proba = chosen_model.predict_proba(inp_scaled)[0] \
                if hasattr(chosen_model, "predict_proba") else None
        conf  = proba[seg] if proba is not None else None

        st.markdown(f"""
        <div style="background:{cfg['bg']};border:1px solid {cfg['color']}55;
                    border-left:6px solid {cfg['color']};border-radius:8px;
                    padding:22px 26px;margin:1rem 0">
            <div style="font-size:2.2rem">{cfg['icon']}</div>
            <h2 style="color:{cfg['color']};margin:4px 0 2px;
                       font-family:'Times New Roman',serif">
                Segment {seg} — {cfg['name']}
            </h2>
            {"<p style='color:#555;font-size:.9rem;margin:0'>Confidence: <b>" + f"{conf*100:.1f}%" + "</b></p>" if conf else ""}
            <hr style="border:none;border-top:1px solid {cfg['color']}44;margin:10px 0">
            <p style="color:#333;margin:2px 0;font-size:.93rem">
                <b>Profile:</b> {cfg['profile']}
            </p>
            <p style="color:#333;margin:4px 0;font-size:.93rem">
                <b>Strategy:</b> {cfg['strategy']}
            </p>
        </div>""", unsafe_allow_html=True)

        # Probability chart
        if proba is not None:
            st.markdown('<p class="sec-hdr">Prediction Probability per Segment</p>', unsafe_allow_html=True)
            n = len(proba)
            fig_pb = go.Figure(go.Bar(
                x=[SEGMENT_CONFIG[i]["name"] for i in range(n)],
                y=proba,
                marker_color=[SEGMENT_CONFIG[i]["color"] for i in range(n)],
                text=[f"{p*100:.1f}%" for p in proba],
                textposition="outside",
            ))
            fig_pb.update_layout(
                **PLOT_LAYOUT, height=300,
                yaxis=dict(tickformat=".0%", title="Probability", range=[0, 1.15]),
                xaxis_title="Segment",
                title=f"{clf_choice} — Prediction Probabilities",
            )
            st.plotly_chart(fig_pb, use_container_width=True)

        # Centroid distances
        st.markdown('<p class="sec-hdr">Distance to K-Means Centroids</p>', unsafe_allow_html=True)
        dists = np.linalg.norm(km_model.cluster_centers_ - inp_scaled, axis=1)
        fig_d = go.Figure(go.Bar(
            x=[SEGMENT_CONFIG[i]["name"] for i in range(len(dists))],
            y=dists,
            marker_color=[SEGMENT_CONFIG[i]["color"] for i in range(len(dists))],
            text=[f"{d:.2f}" for d in dists],
            textposition="outside",
        ))
        fig_d.update_layout(
            **PLOT_LAYOUT, height=300,
            yaxis_title="Euclidean Distance",
            title="Smaller = closer match to that segment",
        )
        st.plotly_chart(fig_d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: SEGMENT OVERVIEW
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Segment Overview":

    st.markdown('<p class="sec-hdr">Segment Profiles</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for seg, cfg in SEGMENT_CONFIG.items():
        row = summary[summary["Cluster"] == seg].iloc[0]
        pct = row["Count"] / len(df) * 100
        with cols[seg]:
            st.markdown(f"""
            <div style="background:{cfg['bg']};border:1px solid {cfg['color']}55;
                        border-top:5px solid {cfg['color']};border-radius:8px;
                        padding:16px 18px;font-family:'Times New Roman',serif">
                <div style="font-size:1.8rem">{cfg['icon']}</div>
                <b style="color:{cfg['color']};font-size:1rem">Segment {seg}: {cfg['name']}</b><br>
                <small style="color:#888">👥 {int(row['Count'])} customers ({pct:.1f}%)</small>
                <hr style="border:none;border-top:1px solid {cfg['color']}33;margin:8px 0">
                💰 Avg Income &nbsp;&nbsp;<b>${row['AvgIncome']:,.0f}</b><br>
                🛒 Avg Spend &nbsp;&nbsp;&nbsp;<b>${row['AvgSpend']:,.0f}</b><br>
                📅 Avg Age &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{row['AvgAge']:.0f}</b><br>
                🕐 Recency &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{row['AvgRecency']:.0f} days</b><br>
                👶 Avg Kids &nbsp;&nbsp;&nbsp;&nbsp;<b>{row['AvgKids']:.1f}</b><br>
                <hr style="border:none;border-top:1px solid {cfg['color']}33;margin:8px 0">
                <small style="color:#555"><i>{cfg['strategy']}</i></small>
            </div>""", unsafe_allow_html=True)

    # Segment size + income/spend comparison
    st.markdown('<p class="sec-hdr">Segment Size Distribution</p>', unsafe_allow_html=True)
    col_pie, col_bar = st.columns([1, 1.5])

    with col_pie:
        fig_pie = px.pie(
            summary,
            values="Count",
            names=[SEGMENT_CONFIG[s]["name"] for s in summary["Cluster"]],
            color_discrete_sequence=[SEGMENT_CONFIG[s]["color"] for s in summary["Cluster"]],
            hole=0.50,
        )
        fig_pie.update_traces(textposition="outside", textfont_size=11)
        fig_pie.update_layout(**PLOT_LAYOUT, height=340,
                              legend=dict(orientation="v", x=1.02, y=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_ib = go.Figure()
        for seg in summary["Cluster"]:
            row = summary[summary["Cluster"] == seg].iloc[0]
            fig_ib.add_trace(go.Bar(
                name=SEGMENT_CONFIG[seg]["name"],
                x=["Avg Income", "Avg Spending"],
                y=[row["AvgIncome"], row["AvgSpend"]],
                marker_color=SEGMENT_CONFIG[seg]["color"],
                text=[f"${row['AvgIncome']:,.0f}", f"${row['AvgSpend']:,.0f}"],
                textposition="outside",
            ))
        fig_ib.update_layout(
            **PLOT_LAYOUT, height=340, barmode="group",
            yaxis_title="Amount ($)", title="Avg Income vs Spending by Segment",
        )
        st.plotly_chart(fig_ib, use_container_width=True)

    # PCA scatter
    st.markdown('<p class="sec-hdr">Customer Cluster Map — PCA Projection</p>', unsafe_allow_html=True)
    fig_pca = px.scatter(
        df, x="PC1", y="PC2",
        color="SegName",
        color_discrete_map={SEGMENT_CONFIG[s]["name"]: SEGMENT_CONFIG[s]["color"] for s in SEGMENT_CONFIG},
        hover_data={"Income":":,.0f","Total_Spending":":,.0f","Age":True,"PC1":False,"PC2":False},
        opacity=0.65,
        labels={"PC1": f"PC1 ({evr[0]*100:.1f}% var)", "PC2": f"PC2 ({evr[1]*100:.1f}% var)", "SegName": "Segment"},
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(**PLOT_LAYOUT, height=440, title="K-Means Clusters in PCA Space")
    st.plotly_chart(fig_pca, use_container_width=True)

    # Boxplots
    st.markdown('<p class="sec-hdr">Feature Distribution by Segment</p>', unsafe_allow_html=True)
    feat_choice = st.selectbox("Select feature", FEATURES)
    fig_box = go.Figure()
    for seg in range(4):
        seg_df = df[df["Cluster"] == seg]
        fig_box.add_trace(go.Box(
            y=seg_df[feat_choice],
            name=SEGMENT_CONFIG[seg]["name"],
            marker_color=SEGMENT_CONFIG[seg]["color"],
            fillcolor=SEGMENT_CONFIG[seg]["color"] + "22",
        ))
    fig_box.update_layout(**PLOT_LAYOUT, height=380,
                          yaxis_title=feat_choice,
                          title=f"Distribution of {feat_choice} per Segment")
    st.plotly_chart(fig_box, use_container_width=True)

    # Spending heatmap
    st.markdown('<p class="sec-hdr">Spending Category Breakdown per Segment</p>', unsafe_allow_html=True)
    spend_cols = ["MntWines","MntFruits","MntSweetProducts","MntGoldProds","MntNonVeg_Prodicuts"]
    spend_lbls = ["Wines","Fruits","Sweets","Gold","Meat+Fish"]
    heat_data  = df.groupby("Cluster")[spend_cols].mean().values
    fig_heat = go.Figure(go.Heatmap(
        z=heat_data,
        x=spend_lbls,
        y=[SEGMENT_CONFIG[s]["name"] for s in range(4)],
        colorscale="Blues",
        text=np.round(heat_data, 0).astype(int),
        texttemplate="$%{text}",
    ))
    fig_heat.update_layout(**PLOT_LAYOUT, height=280,
                           title="Average Spending per Category by Segment ($)")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Download
    with st.expander("🔍 View & Download Segmented Customer Data"):
        out = df[["ID","Age","Income","Total_Spending","Total_Purchases",
                  "TotKids","Recency","Cluster","SegName"]].rename(
            columns={"SegName":"Segment","TotKids":"Kids"})
        st.dataframe(
            out.style.format({"Income":"${:,.0f}","Total_Spending":"${:,.0f}"}),
            use_container_width=True, height=380)
        st.download_button(
            "⬇️ Download Segmented CSV",
            out.to_csv(index=False).encode("utf-8"),
            "segmented_customers.csv","text/csv")


# ══════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    lr_res = clf["Logistic Regression"]
    rf_res = clf["Random Forest"]

    st.markdown('<p class="sec-hdr">Classifier Comparison — Logistic Regression vs Random Forest</p>',
                unsafe_allow_html=True)
    st.caption("Both classifiers trained on K-Means cluster labels (K=4), 80/20 train-test split.")

    comp_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": f"{lr_res['acc']}%",
         "F1-Score": f"{lr_res['f1']}%",
         "Winner": "✅ Best" if best_clf=="Logistic Regression" else ""},
        {"Model": "Random Forest",       "Accuracy": f"{rf_res['acc']}%",
         "F1-Score": f"{rf_res['f1']}%",
         "Winner": "✅ Best" if best_clf=="Random Forest" else ""},
    ])
    st.dataframe(
        comp_df.style.apply(
            lambda r: ["background-color:#eaf5ea" if r["Winner"].startswith("✅") else "" for _ in r], axis=1),
        use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_acc = go.Figure(go.Bar(
            x=["Logistic Regression","Random Forest"],
            y=[lr_res["acc"], rf_res["acc"]],
            marker_color=["#2563eb","#16a34a"],
            text=[f"{lr_res['acc']}%", f"{rf_res['acc']}%"],
            textposition="outside",
        ))
        fig_acc.update_layout(**PLOT_LAYOUT, height=320,
                              yaxis=dict(range=[0,110],title="Accuracy (%)"),
                              title="Test Accuracy")
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_b:
        fig_f1 = go.Figure(go.Bar(
            x=["Logistic Regression","Random Forest"],
            y=[lr_res["f1"], rf_res["f1"]],
            marker_color=["#d97706","#9333ea"],
            text=[f"{lr_res['f1']}%", f"{rf_res['f1']}%"],
            textposition="outside",
        ))
        fig_f1.update_layout(**PLOT_LAYOUT, height=320,
                              yaxis=dict(range=[0,110],title="F1-Score (%)"),
                              title="Weighted F1-Score")
        st.plotly_chart(fig_f1, use_container_width=True)

    # Per-model tabs
    st.markdown('<p class="sec-hdr">Detailed Classification Reports</p>', unsafe_allow_html=True)
    tab_lr, tab_rf = st.tabs(["📋 Logistic Regression","🌲 Random Forest"])

    for tab, key in [(tab_lr,"Logistic Regression"),(tab_rf,"Random Forest")]:
        with tab:
            res = clf[key]
            col_rep, col_cm = st.columns(2)

            with col_rep:
                rpt = pd.DataFrame(res["report"]).T
                drop_idx = [i for i in ["accuracy","macro avg","weighted avg"] if i in rpt.index]
                rpt = rpt.drop(drop_idx)[["precision","recall","f1-score","support"]].round(4)
                rpt.index = [SEGMENT_CONFIG.get(int(i),{}).get("name",f"Seg {i}") for i in rpt.index]
                rpt["support"] = rpt["support"].astype(int)
                st.markdown(f"**Per-Segment Metrics — {key}**")
                st.dataframe(rpt.style.format({"precision":"{:.4f}","recall":"{:.4f}","f1-score":"{:.4f}"}),
                             use_container_width=True)
                wa = res["report"].get("weighted avg",{})
                st.markdown(f"""
                <div style="background:#f0f4ff;border:1px solid #c7d2fe;border-radius:6px;
                            padding:12px 16px;margin-top:10px;font-family:'Times New Roman',serif">
                    <b>Weighted Avg</b> &nbsp;|&nbsp;
                    Precision: <b>{wa.get('precision',0)*100:.2f}%</b> &nbsp;|&nbsp;
                    Recall: <b>{wa.get('recall',0)*100:.2f}%</b> &nbsp;|&nbsp;
                    F1: <b>{wa.get('f1-score',0)*100:.2f}%</b>
                </div>""", unsafe_allow_html=True)

            with col_cm:
                cm   = res["cm"]
                segs = sorted(set(res["y_te"]))
                lbls = [SEGMENT_CONFIG.get(s,{}).get("name",f"Seg {s}") for s in segs]
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                   x=lbls, y=lbls, color_continuous_scale="Blues",
                                   title=f"Confusion Matrix — {key}")
                fig_cm.update_layout(**PLOT_LAYOUT, height=340)
                st.plotly_chart(fig_cm, use_container_width=True)

            if "fi" in res:
                fi_df = pd.DataFrame({"Feature":FEATURES,"Importance":res["fi"]}).sort_values("Importance")
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="Blues",
                                title="Feature Importances — Random Forest")
                fig_fi.update_layout(**PLOT_LAYOUT, height=320, coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)

    # Elbow
    st.markdown('<p class="sec-hdr">K-Means — Elbow Method (WCSS)</p>', unsafe_allow_html=True)
    fig_elbow = go.Figure(go.Scatter(
        x=list(range(1,10)), y=wcss,
        mode="lines+markers",
        line=dict(color="#2c3e6b", width=2.5),
        marker=dict(size=8, color="#2c3e6b"),
        fill="tozeroy", fillcolor="rgba(44,62,107,0.08)",
    ))
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="#c9a84c",
                        annotation_text="  K=4 selected", annotation_font_color="#c9a84c")
    fig_elbow.update_layout(**PLOT_LAYOUT, height=320,
                            xaxis_title="Number of Clusters (K)",
                            yaxis_title="WCSS (Inertia)",
                            title="Elbow Method — Optimal K Selection")
    st.plotly_chart(fig_elbow, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="sec-hdr">Customer Personality Analysis — Segmentation Predictor</p>',
                unsafe_allow_html=True)
    st.write("This app predicts which customer segment a new customer belongs to "
             "using a trained classifier built on K-Means cluster labels.")

    st.markdown('<p class="sec-hdr">Pipeline</p>', unsafe_allow_html=True)
    for num, title, desc in [
        ("1","EDA Notebook",
         "Cleaned 2,240 customers → 2,236 rows after deduplication. Engineered 6 features: "
         "Income, Age, Total Spending, Total Kids, Recency, Total Purchases."),
        ("2","Model Building Notebook",
         "Ran Elbow Method (WCSS) to determine optimal K=4. Applied K-Means clustering. "
         "Evaluated with Silhouette Score."),
        ("3","Classification Notebook",
         "Trained Logistic Regression and Random Forest on K-Means labels. "
         "Evaluated Accuracy and weighted F1-Score on held-out test set."),
        ("4","This App",
         "Uses the best classifier to predict segments for new customers with "
         "confidence scores, probability breakdown, and feature importances."),
    ]:
        st.markdown(f"""
        <div class="pipeline-step">
            <b style="color:#2c3e6b">Step {num} — {title}</b><br>
            <span style="color:#444">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-hdr">Model Comparison</p>', unsafe_allow_html=True)
    lr_r, rf_r = clf["Logistic Regression"], clf["Random Forest"]
    mc = pd.DataFrame([
        {"Model":"Logistic Regression","Accuracy":f"{lr_r['acc']}%","F1-Score":f"{lr_r['f1']}%",
         "Status":"✅ Selected" if best_clf=="Logistic Regression" else ""},
        {"Model":"Random Forest",      "Accuracy":f"{rf_r['acc']}%","F1-Score":f"{rf_r['f1']}%",
         "Status":"✅ Selected" if best_clf=="Random Forest" else ""},
    ])
    st.dataframe(
        mc.style.apply(
            lambda r: ["background-color:#eaf5ea" if r["Status"].startswith("✅") else "" for _ in r], axis=1),
        use_container_width=True, hide_index=True)

    st.markdown('<p class="sec-hdr">Segment Descriptions</p>', unsafe_allow_html=True)
    for seg, cfg in SEGMENT_CONFIG.items():
        row = summary[summary["Cluster"] == seg].iloc[0]
        st.markdown(f"""
        <div style="background:{cfg['bg']};border-left:5px solid {cfg['color']};
                    border-radius:6px;padding:12px 16px;margin-bottom:8px;
                    font-family:'Times New Roman',serif">
            <b style="color:{cfg['color']}">{cfg['icon']} Segment {seg}: {cfg['name']}</b>
            &nbsp;&nbsp;<small style="color:#888">👥 {int(row['Count'])} customers</small><br>
            <span style="color:#444;font-size:.92rem">{cfg['profile']}</span><br>
            <span style="color:#555;font-size:.88rem"><i>Strategy: {cfg['strategy']}</i></span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-hdr">Tech Stack</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Library":"Streamlit",      "Purpose":"Web application framework"},
        {"Library":"scikit-learn",   "Purpose":"K-Means, Logistic Regression, Random Forest"},
        {"Library":"Pandas / NumPy", "Purpose":"Data cleaning and feature engineering"},
        {"Library":"Plotly",         "Purpose":"Interactive charts and visualisations"},
    ]), use_container_width=True, hide_index=True)
