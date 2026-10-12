"""
NovaGen Risk Studio
Streamlit app around the five models from novagen.ipynb
(Logistic Regression, KNN, Random Forest, Gradient Boosting, Voting Classifier).

Run:  streamlit run novagen_app.py
Needs novagen_dataset.csv in the same folder.
Educational project - not a medical device.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "novagen_dataset.csv"

TEAL, BLUE, ROSE, AMBER, GREEN = "#0D9488", "#2563EB", "#E11D48", "#F59E0B", "#16A34A"

NUM = ["Age", "BMI", "Blood_Pressure", "Cholesterol", "Glucose_Level", "Heart_Rate",
       "Sleep_Hours", "Exercise_Hours", "Water_Intake", "Stress_Level"]
ORD = ["Smoking", "Alcohol", "Diet", "MentalHealth", "PhysicalActivity",
       "MedicalHistory", "Allergies"]
BOOL = ["Diet_Type__Vegan", "Diet_Type__Vegetarian",
        "Blood_Group_AB", "Blood_Group_B", "Blood_Group_O"]
GROUPS = {**{c: [c] for c in NUM + ORD},
          "Diet_Type": ["Diet_Type__Vegan", "Diet_Type__Vegetarian"],
          "Blood_Group": ["Blood_Group_AB", "Blood_Group_B", "Blood_Group_O"]}
NICE = {"Blood_Pressure": "Blood pressure", "Glucose_Level": "Glucose level",
        "Heart_Rate": "Heart rate", "Sleep_Hours": "Sleep hours",
        "Exercise_Hours": "Exercise hours", "Water_Intake": "Water intake",
        "Stress_Level": "Stress level", "MentalHealth": "Mental health",
        "PhysicalActivity": "Physical activity", "MedicalHistory": "Medical history",
        "Diet_Type": "Diet type", "Blood_Group": "Blood group"}
nice = lambda c: NICE.get(c, c)

MODEL_NAMES = ["Random Forest", "Gradient Boosting", "Voting Classifier",
               "KNN", "Logistic Regression"]

st.set_page_config(page_title="NovaGen Risk Studio", page_icon="🧬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hero {background: linear-gradient(135deg,#0F766E 0%,#2563EB 100%); padding: 1.5rem 1.8rem;
       border-radius: 18px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff; padding: 0;}
.hero p {margin: .35rem 0 0; opacity: .92; font-size: 1.02rem;}
.kpi {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: .85rem 1rem;
      background: rgba(128,128,128,.07); height: 100%;}
.kpi .l {opacity: .7; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em;}
.kpi .v {font-size: 1.75rem; font-weight: 700; line-height: 1.25;}
.kpi .s {opacity: .65; font-size: .8rem;}
.verdict {border-radius: 14px; padding: 1rem 1.2rem; margin: .1rem 0 .6rem; color: #fff;}
.verdict h3 {margin: 0; color: #fff; font-size: 1.3rem; padding: 0;}
.verdict p {margin: .3rem 0 0; opacity: .95;}
.insight {border-left: 4px solid #0D9488; background: rgba(13,148,136,.10);
          padding: .65rem 1rem; border-radius: 0 10px 10px 0; margin: .4rem 0; font-size: .95rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================ data + models
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("novagen_dataset.csv not found. Put it in the same folder as this file.")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training the 5 models from your notebook (about 10 seconds)…")
def train():
    df = load_data()
    X, y = df.drop("Target", axis=1), df["Target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaled = lambda m: Pipeline([("scaler", StandardScaler()), ("model", m)])
    models = {
        "Logistic Regression": scaled(LogisticRegression(penalty="l2", solver="liblinear",
                                                         max_iter=1000)),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None,
                                                random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                        max_depth=3, random_state=42),
        "Voting Classifier": scaled(VotingClassifier(
            estimators=[("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
                        ("knn", KNeighborsClassifier(n_neighbors=5)),
                        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))],
            voting="soft")),
    }
    proba, rows = {}, []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        proba[name] = m.predict_proba(X_te)[:, 1]
        pred = m.predict(X_te)
        rows.append(dict(Model=name, Accuracy=accuracy_score(y_te, pred),
                         Recall=recall_score(y_te, pred), Precision=precision_score(y_te, pred),
                         F1=f1_score(y_te, pred), AUC=roc_auc_score(y_te, proba[name])))
    baseline = X.astype(float).median()
    baseline[BOOL] = 0.0   # typical patient: diet 'Standard', blood group 'A'
    return dict(models=models, proba=proba, X=X, y=y, X_te=X_te, y_te=y_te,
                metrics=pd.DataFrame(rows).set_index("Model").loc[MODEL_NAMES],
                baseline=baseline, n_train=len(X_tr), n_test=len(X_te),
                importance={n: pd.Series(models[n].feature_importances_, X.columns)
                            for n in ["Random Forest", "Gradient Boosting"]})


def metrics_at(y_true, proba, thr):
    pred = (proba > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y_true, pred), rec=recall_score(y_true, pred),
                prec=precision_score(y_true, pred, zero_division=0),
                f1=f1_score(y_true, pred), tn=tn, fp=fp, fn=fn, tp=tp, pred=pred)


def risk_band(p, thr):
    if p > thr:
        return "High risk", ROSE
    if p > thr - 0.15:
        return "Borderline", AMBER
    return "Low risk", GREEN


def kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def drivers(art, model, row: pd.DataFrame) -> pd.Series:
    """Occlusion explanation: swap each feature group for the typical-patient value and
    see how much the risk score moves. Positive = this patient's value adds risk."""
    names = list(GROUPS)
    batch = pd.concat([row] * len(names), ignore_index=True)
    for i, g in enumerate(names):
        for c in GROUPS[g]:
            batch.loc[i, c] = bool(art["baseline"][c]) if c in BOOL else art["baseline"][c]
    batch[BOOL] = batch[BOOL].astype(bool)
    p_swapped = model.predict_proba(batch[art["X"].columns])[:, 1]
    p_now = model.predict_proba(row[art["X"].columns])[0, 1]
    return pd.Series(p_now - p_swapped, index=[nice(n) for n in names])


art = train()
df = load_data()
X, y = art["X"], art["y"]

# ============================================================ sidebar
with st.sidebar:
    st.markdown("## 🧬 NovaGen")
    page = st.radio("Navigate", ["🏠 Data explorer", "🩺 Risk assessment",
                                 "📈 Model comparison", "📦 Batch screening"],
                    label_visibility="collapsed")
    st.divider()
    active = st.selectbox("Active model", MODEL_NAMES, 0,
                          help="Used on the assessment, comparison and batch pages. "
                               "Random Forest had the best recall in your notebook.")
    am = art["metrics"].loc[active]
    st.write(f"Recall **{am['Recall']:.1%}** · Accuracy **{am['Accuracy']:.1%}**")
    st.caption(f"{len(df):,} records · {art['n_train']:,} train / {art['n_test']:,} test")
    st.divider()
    st.caption("⚠️ Educational project. Predictions come from a statistical model trained on "
               "this dataset and are not medical advice.")
model = art["models"][active]

st.markdown("""<div class="hero"><h1>NovaGen Risk Studio</h1>
<p>Explore the data, screen a patient with five machine-learning models, and see what drives each prediction.</p></div>""",
            unsafe_allow_html=True)

# ============================================================ PAGE: explorer
if page == "🏠 Data explorer":
    c1, c2 = st.columns(2)
    age_rng = c1.slider("Age range", 0, 100, (0, 100))
    groups = c2.multiselect("Blood group", ["A", "B", "AB", "O"], default=["A", "B", "AB", "O"])
    bg = np.select([df["Blood_Group_AB"], df["Blood_Group_B"], df["Blood_Group_O"]],
                   ["AB", "B", "O"], default="A")
    d = df[df["Age"].between(*age_rng) & pd.Series(bg, index=df.index).isin(groups)].copy()
    if d.empty:
        st.warning("No records match these filters.")
        st.stop()
    d["Class"] = d["Target"].map({1: "Target = 1 (high-risk class)", 0: "Target = 0"})

    k = st.columns(4)
    k[0].markdown(kpi("Patients", f"{len(d):,}", "in current filter"), unsafe_allow_html=True)
    k[1].markdown(kpi("High-risk class", f"{d['Target'].mean():.1%}",
                      f"{d['Target'].sum():,} patients"), unsafe_allow_html=True)
    k[2].markdown(kpi("Average age", f"{d['Age'].mean():.1f}", "years"), unsafe_allow_html=True)
    k[3].markdown(kpi("Avg blood pressure", f"{d['Blood_Pressure'].mean():.0f}",
                      f"BMI {d['BMI'].mean():.1f}"), unsafe_allow_html=True)
    st.write("")

    a, b = st.columns(2)
    d["Age band"] = pd.cut(d["Age"], [-1, 17, 34, 49, 64, 200],
                           labels=["0–17", "18–34", "35–49", "50–64", "65+"])
    ab = d.groupby("Age band", observed=True)["Target"].agg(["mean", "size"]).reset_index()
    fig = px.bar(ab, x="Age band", y="mean", custom_data=["size"], title="High-risk share by age band",
                 color_discrete_sequence=[TEAL])
    fig.update_traces(hovertemplate="%{x}<br>%{y:.1%} high-risk<br>%{customdata[0]:,} patients")
    fig.update_yaxes(tickformat=".0%", title=None, range=[0, 1])
    fig.update_xaxes(title=None)
    a.plotly_chart(fig, width="stretch")

    corr = d[NUM + ORD + BOOL + ["Target"]].astype(float).corr()["Target"].drop("Target")
    corr.index = [nice(i) for i in corr.index]
    corr = corr.reindex(corr.abs().sort_values().index).tail(12)
    fig = px.bar(corr, x=corr.values, y=corr.index, orientation="h",
                 title="Correlation with Target (top 12)", color=corr.values,
                 color_continuous_scale=[ROSE, "#CBD5E1", TEAL], range_color=[-0.5, 0.5])
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    b.plotly_chart(fig, width="stretch")

    st.markdown("##### Compare a measurement across the two classes")
    f1c, f2c = st.columns([1, 3])
    feat = f1c.selectbox("Measurement", NUM, index=NUM.index("BMI"), format_func=nice)
    fig = px.histogram(d, x=feat, color="Class", barmode="overlay", nbins=40, opacity=0.65,
                       marginal="box", color_discrete_map={"Target = 1 (high-risk class)": ROSE,
                                                           "Target = 0": TEAL})
    fig.update_layout(legend_title=None, legend=dict(orientation="h", y=1.12))
    f2c.plotly_chart(fig, width="stretch")

    st.markdown("##### Lifestyle & history levels vs. outcome")
    f1c, f2c = st.columns([1, 3])
    o = f1c.selectbox("Factor (levels 0–2)", ORD, format_func=nice)
    ob = d.groupby(o)["Target"].agg(["mean", "size"]).reset_index()
    fig = px.bar(ob, x=o, y="mean", custom_data=["size"], color_discrete_sequence=[BLUE])
    fig.update_traces(hovertemplate="Level %{x}<br>%{y:.1%} high-risk<br>%{customdata[0]:,} patients")
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="High-risk share")
    fig.update_xaxes(type="category", title=f"{nice(o)} level")
    fig.update_layout(height=300)
    f2c.plotly_chart(fig, width="stretch")

    top = corr.abs().idxmax()
    st.markdown(f'<div class="insight">Strongest single relationship with the target in this view: '
                f'<b>{top}</b> (correlation {corr[top]:+.2f}).</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">The 0–2 lifestyle and history factors show almost no '
                'relationship with the target on their own, so the models rely mostly on the '
                'clinical measurements.</div>', unsafe_allow_html=True)
    with st.expander("Correlation heatmap (measurements)"):
        cm = d[NUM + ["Target"]].corr()
        fig = px.imshow(cm.rename(index=nice, columns=nice), text_auto=".2f", aspect="auto",
                        color_continuous_scale=[ROSE, "#F8FAFC", TEAL], zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")

# ============================================================ PAGE: assessment
elif page == "🩺 Risk assessment":
    X_te, y_te = art["X_te"], art["y_te"]
    if "seed" not in st.session_state:
        st.session_state.seed = 0

    top = st.columns([3, 1, 2])
    preset = top[0].radio("Start from", ["Typical patient (dataset median)",
                                         "Real patient labelled high-risk",
                                         "Real patient labelled low-risk"], horizontal=True)
    if top[1].button("🎲 Shuffle patient", disabled=preset.startswith("Typical")):
        st.session_state.seed += 1
    thr = top[2].slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                        help="Risk score above this is flagged HIGH RISK. Lower it to catch more "
                             "high-risk patients (higher recall) at the cost of more false alarms.")

    if preset.startswith("Typical"):
        P, true_label = art["baseline"].copy(), None
    else:
        want = 1 if "high-risk" in preset else 0
        pool = X_te[y_te == want]
        P = pool.iloc[np.random.default_rng(st.session_state.seed).integers(len(pool))]
        true_label = want
    P = P.copy()
    kx = lambda n: f"{n}|{preset}|{st.session_state.seed}"
    if true_label is not None:
        st.caption(f"Loaded a real test-set record. Recorded label: "
                   f"**{'high-risk (Target = 1)' if true_label else 'low-risk (Target = 0)'}**.")

    rng = {c: (int(X[c].min()), int(X[c].max())) for c in NUM}
    left, right = st.columns([3, 2], gap="large")
    vals = {}
    with left:
        with st.container(border=True):
            st.markdown("**Vitals & demographics**")
            cols = st.columns(2)
            for i, c in enumerate(NUM):
                lo, hi = rng[c]
                vals[c] = cols[i % 2].slider(nice(c), lo, hi, int(np.clip(P[c], lo, hi)),
                                             key=kx(c), help=f"Training range {lo}–{hi}")
        with st.container(border=True):
            st.markdown("**Lifestyle & history** (encoded level 0–2, as in the dataset)")
            cols = st.columns(4)
            for i, c in enumerate(ORD):
                vals[c] = cols[i % 4].select_slider(nice(c), [0, 1, 2], int(P[c]), key=kx(c))
        with st.container(border=True):
            st.markdown("**Diet & blood group**")
            a, b = st.columns(2)
            diet_opts = ["Standard", "Vegetarian", "Vegan"]
            d0 = 2 if P["Diet_Type__Vegan"] else 1 if P["Diet_Type__Vegetarian"] else 0
            diet = a.radio("Diet type", diet_opts, d0, horizontal=True, key=kx("diet"))
            bg_opts = ["A", "B", "AB", "O"]
            b0 = 2 if P["Blood_Group_AB"] else 1 if P["Blood_Group_B"] else 3 if P["Blood_Group_O"] else 0
            bgv = b.radio("Blood group", bg_opts, b0, horizontal=True, key=kx("bg"))
        vals.update({"Diet_Type__Vegan": diet == "Vegan", "Diet_Type__Vegetarian": diet == "Vegetarian",
                     "Blood_Group_AB": bgv == "AB", "Blood_Group_B": bgv == "B",
                     "Blood_Group_O": bgv == "O"})

    row = pd.DataFrame([vals])[X.columns]
    row[BOOL] = row[BOOL].astype(bool)
    p = float(model.predict_proba(row)[0, 1])
    label, colr = risk_band(p, thr)
    msg = {"High risk": "Flag for follow-up screening.",
           "Borderline": "Close to the alert threshold. Consider monitoring.",
           "Low risk": "Below the alert threshold."}[label]

    with right:
        st.markdown(f'<div class="verdict" style="background:{colr}"><h3>{label}</h3>'
                    f'<p>{msg} ({active})</p></div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".0f"},
            title={"text": "Risk score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": colr},
                   "steps": [{"range": [0, thr * 100], "color": "rgba(22,163,74,.15)"},
                             {"range": [thr * 100, 100], "color": "rgba(225,29,72,.15)"}],
                   "threshold": {"line": {"color": "#111", "width": 4}, "thickness": .8,
                                 "value": thr * 100}}))
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=0))
        st.plotly_chart(fig, width="stretch")

        allp = pd.Series({n: float(art["models"][n].predict_proba(row)[0, 1]) for n in MODEL_NAMES})
        flags = int((allp > thr).sum())
        st.markdown(f"**Model consensus:** {flags} of 5 models flag high risk")
        fig = px.bar(allp.sort_values(), x=allp.sort_values().values, y=allp.sort_values().index,
                     orientation="h", color=(allp.sort_values() > thr).map({True: "High", False: "Not high"}),
                     color_discrete_map={"High": ROSE, "Not high": TEAL})
        fig.add_vline(x=thr, line_dash="dash", line_color="#111")
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title=None)
        fig.update_yaxes(title=None)
        fig.update_layout(height=230, showlegend=False, margin=dict(l=0, r=10, t=5, b=0))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.markdown("##### What is moving this patient's score?")
        st.caption("Each bar shows how much the score changes when that factor is swapped for the "
                   "typical-patient value. Red = pushes risk up, green = pushes it down.")
        dv = drivers(art, model, row)
        dv = dv.reindex(dv.abs().sort_values().index).tail(8)
        fig = px.bar(dv, x=dv.values, y=dv.index, orientation="h",
                     color=(dv > 0).map({True: "Raises risk", False: "Lowers risk"}),
                     color_discrete_map={"Raises risk": ROSE, "Lowers risk": GREEN})
        fig.update_xaxes(tickformat="+.0%", title="Change in risk score")
        fig.update_yaxes(title=None)
        fig.update_layout(height=340, legend_title=None, legend=dict(orientation="h", y=1.12),
                          margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with w2:
        st.markdown("##### What-if: vary one measurement")
        feat = st.selectbox("Measurement", NUM, index=NUM.index("BMI"), format_func=nice,
                            label_visibility="collapsed")
        lo, hi = rng[feat]
        xs = np.arange(lo, hi + 1)
        grid = pd.concat([row] * len(xs), ignore_index=True)
        grid[feat] = xs
        grid[BOOL] = grid[BOOL].astype(bool)
        ys = model.predict_proba(grid[X.columns])[:, 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=3)))
        fig.add_hline(y=thr, line_dash="dash", line_color=ROSE, annotation_text="threshold")
        fig.add_trace(go.Scatter(x=[vals[feat]], y=[p], mode="markers",
                                 marker=dict(size=13, color="black", symbol="diamond")))
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Risk score")
        fig.update_xaxes(title=nice(feat))
        fig.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

# ============================================================ PAGE: comparison
elif page == "📈 Model comparison":
    mt = art["metrics"]
    st.markdown("##### All five models on the held-out test set (threshold 0.50, as in your notebook)")
    tbl = (mt * 100).reset_index()
    cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.1f%%")
           for c in ["Accuracy", "Recall", "Precision", "F1", "AUC"]}
    st.dataframe(tbl, hide_index=True, width="stretch", column_config=cfg)
    best = mt["Recall"].idxmax()
    st.markdown(f'<div class="insight">In a screening setting, missing a high-risk patient is the '
                f'costly error, so recall is the key metric. <b>{best}</b> has the highest recall '
                f'({mt.loc[best, "Recall"]:.1%}) with {mt.loc[best, "Accuracy"]:.1%} accuracy.</div>',
                unsafe_allow_html=True)

    long = mt[["Accuracy", "Recall", "Precision", "F1"]].reset_index().melt("Model", var_name="Metric")
    fig = px.bar(long, x="Model", y="value", color="Metric", barmode="group",
                 color_discrete_sequence=[TEAL, ROSE, BLUE, AMBER])
    fig.update_yaxes(range=[0.6, 1], tickformat=".0%", title=None)
    fig.update_xaxes(title=None)
    fig.update_layout(height=360, legend_orientation="h", legend_y=1.12, legend_title=None)
    st.plotly_chart(fig, width="stretch")

    st.markdown(f"##### Drill into **{active}**")
    y_te, proba = art["y_te"], art["proba"][active]
    thr = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                    help="Slide left to catch more high-risk patients; watch false alarms rise.")
    m = metrics_at(y_te, proba, thr)
    k = st.columns(5)
    k[0].markdown(kpi("Recall", f"{m['rec']:.1%}", "high-risk caught"), unsafe_allow_html=True)
    k[1].markdown(kpi("Missed high-risk", f"{m['fn']:,}", "false negatives"), unsafe_allow_html=True)
    k[2].markdown(kpi("False alarms", f"{m['fp']:,}", "false positives"), unsafe_allow_html=True)
    k[3].markdown(kpi("Precision", f"{m['prec']:.1%}", "flags that are right"), unsafe_allow_html=True)
    k[4].markdown(kpi("Accuracy", f"{m['acc']:.1%}", f"{art['n_test']:,} test patients"),
                  unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    cmx = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = px.imshow(cmx, text_auto=True, color_continuous_scale="Teal",
                    x=["Predicted low", "Predicted high"], y=["Actual low", "Actual high"],
                    title="Confusion matrix")
    fig.update_layout(coloraxis_showscale=False)
    c1.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for n in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y_te, art["proba"][n])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n,
                                 line=dict(width=4 if n == active else 2),
                                 opacity=1 if n == active else 0.5))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="gray", dash="dash")))
    fig.add_trace(go.Scatter(x=[m["fp"] / (m["fp"] + m["tn"])], y=[m["rec"]], mode="markers",
                             name="Your threshold",
                             marker=dict(size=13, color="black", symbol="diamond")))
    fig.update_layout(title="ROC curves (all models)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", legend=dict(font_size=10))
    c2.plotly_chart(fig, width="stretch")

    st.markdown("##### What the tree-based models rely on")
    which = st.radio("Model", list(art["importance"]), horizontal=True, label_visibility="collapsed")
    imp = art["importance"][which].rename(index=nice).sort_values().tail(12)
    fig = px.bar(imp, x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[TEAL])
    fig.update_xaxes(title="Importance")
    fig.update_yaxes(title=None)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

# ============================================================ PAGE: batch
else:
    st.subheader("Screen many patients at once")
    st.write(f"Upload a CSV with the same columns as the training data (a `Target` column is optional). "
             f"Scoring uses the **{active}** model.")
    st.download_button("⬇️ Download a sample template", X.head(5).to_csv(index=False),
                       "novagen_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    thr = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05)
    if up is not None:
        new = pd.read_csv(up)
        missing = [c for c in X.columns if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            data = new[X.columns].copy()
            for c in BOOL:
                data[c] = data[c].astype(str).str.lower().isin(["true", "1", "1.0"])
            try:
                score = model.predict_proba(data)[:, 1]
            except Exception as e:
                st.error(f"Could not score the file: {e}")
                st.stop()
            out = new.copy()
            out["Risk score"] = score.round(3)
            out["Risk level"] = [risk_band(s, thr)[0] for s in score]
            k = st.columns(4)
            k[0].markdown(kpi("Patients screened", f"{len(out):,}"), unsafe_allow_html=True)
            k[1].markdown(kpi("Flagged high risk", f"{(out['Risk level'] == 'High risk').sum():,}",
                              f"{(out['Risk level'] == 'High risk').mean():.1%} of patients"),
                          unsafe_allow_html=True)
            k[2].markdown(kpi("Borderline", f"{(out['Risk level'] == 'Borderline').sum():,}"),
                          unsafe_allow_html=True)
            if "Target" in out.columns:
                rec = recall_score(out["Target"].astype(int), (score > thr).astype(int))
                k[3].markdown(kpi("Recall vs your Target", f"{rec:.1%}"), unsafe_allow_html=True)
            else:
                k[3].markdown(kpi("Average score", f"{score.mean():.2f}"), unsafe_allow_html=True)
            st.write("")
            fig = px.histogram(out, x="Risk score", color="Risk level", nbins=30,
                               color_discrete_map={"High risk": ROSE, "Borderline": AMBER,
                                                   "Low risk": GREEN},
                               title="Distribution of risk scores")
            st.plotly_chart(fig, width="stretch")
            st.markdown("##### Highest-risk patients first")
            st.dataframe(out.sort_values("Risk score", ascending=False), width="stretch",
                         hide_index=True)
            st.download_button("⬇️ Download scored CSV", out.to_csv(index=False),
                               "novagen_screened.csv", "text/csv")
