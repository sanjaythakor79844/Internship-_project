import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2.5rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:1.5rem; }
    .metric-card { background:linear-gradient(135deg,#667eea,#764ba2); padding:1.5rem; border-radius:10px; color:white; text-align:center; margin:0.5rem 0; }
    .prediction-box { background:linear-gradient(135deg,#11998e,#38ef7d); padding:2rem; border-radius:15px; color:white; text-align:center; font-size:2rem; font-weight:bold; margin:2rem 0; }
    .stButton>button { width:100%; background:linear-gradient(135deg,#667eea,#764ba2); color:white; font-size:1.1rem; padding:0.75rem; border-radius:10px; border:none; font-weight:bold; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#1a1a2e,#16213e,#0f3460); }
    [data-testid="stSidebar"] p,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,[data-testid="stSidebar"] span,[data-testid="stSidebar"] div { color:white !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#38ef7d !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color:#aaa !important; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background:rgba(255,255,255,0.07) !important; border:2px solid rgba(255,255,255,0.12) !important;
        border-radius:12px !important; color:white !important; text-align:left !important;
        padding:12px 16px !important; margin:3px 0 !important; font-size:0.95rem !important;
        transition:background 0.25s ease,border 0.25s ease,box-shadow 0.25s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background:linear-gradient(135deg,rgba(102,126,234,0.7),rgba(118,75,162,0.7)) !important;
        border:2px solid #667eea !important;
        box-shadow:0 0 14px rgba(102,126,234,0.55) !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus,
    [data-testid="stSidebar"] [data-testid="stButton"] button:active {
        background:linear-gradient(135deg,#667eea,#764ba2) !important;
        border:2px solid #a78bfa !important;
        box-shadow:0 0 20px rgba(102,126,234,0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── CACHED LOADERS ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        model      = pickle.load(open(os.path.join(BASE, 'best_model.pkl'), 'rb'))
        encoders   = pickle.load(open(os.path.join(BASE, 'encoders.pkl'),   'rb'))
        model_info = pickle.load(open(os.path.join(BASE, 'model_info.pkl'), 'rb'))
        return model, encoders, model_info
    except Exception as e:
        st.error(f"⚠️ Model load error: {e}")
        return None, None, None

@st.cache_data(show_spinner="Loading data...")
def load_data():
    try:
        return pd.read_csv(os.path.join(BASE, 'insurance (1).csv'))
    except Exception as e:
        st.error(f"⚠️ Data load error: {e}")
        return None

@st.cache_data
def get_model_results():
    return pd.DataFrame({
        'Model'        : ['Gradient Boosting','Random Forest','Ridge Regression',
                          'Linear Regression','Lasso Regression','Decision Tree'],
        'Accuracy (%)' : [87.67, 86.93, 86.62, 86.54, 86.55, 72.18],
        'Error ($)'    : [2425, 2349, 2771, 2773, 2772, 2953],
        'RMSE ($)'     : [4375, 4505, 4558, 4571, 4569, 6572],
        'Type'         : ['Ensemble','Ensemble','Linear','Linear','Linear','Tree']
    })

@st.cache_data
def get_analysis_data(df):
    """Pre-compute all analysis data once"""
    df_enc = df.copy()
    df_enc['sex']    = df_enc['sex'].map({'male':1,'female':0})
    df_enc['smoker'] = df_enc['smoker'].map({'yes':1,'no':0})
    df_enc['region'] = df_enc['region'].astype('category').cat.codes
    corr = df_enc.corr()
    smoker_corr = round(corr['charges']['smoker'], 2)

    s_avg  = df[df['smoker']=='yes']['charges'].mean()
    ns_avg = df[df['smoker']=='no']['charges'].mean()

    df_age = df.copy()
    df_age['age_group'] = pd.cut(df_age['age'], bins=[17,25,35,45,55,65],
                                  labels=['18-25','26-35','36-45','46-55','56-64'])
    df_age['bmi_cat']   = pd.cut(df_age['bmi'], bins=[0,18.5,25,30,100],
                                  labels=['Underweight','Normal','Overweight','Obese'])
    age_grp = df_age.groupby('age_group', observed=True)['charges'].mean().reset_index()
    bmi_grp = df_age.groupby('bmi_cat',   observed=True)['charges'].mean().reset_index()

    return corr, smoker_corr, s_avg, ns_avg, age_grp, bmi_grp

# Load everything once at startup
model, encoders, model_info = load_model()
df = load_data()
results_df = get_model_results()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Insurance Predictor")
    st.markdown("*Student ML Project*")
    st.markdown("---")

    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"

    for icon, label in [("🏠","Home"),("📊","Dashboard"),("🔮","Predict"),
                         ("📈","Analysis"),("🤖","Models"),("ℹ️","About")]:
        if st.sidebar.button(f"{icon}  {label}", key=f"{icon} {label}", use_container_width=True):
            st.session_state.page = f"{icon} {label}"

    page = st.session_state.page
    st.markdown("---")

    if model_info:
        st.markdown("### 📈 Model Performance")
        st.metric("Best Model", model_info['model_name'])
        st.metric("Accuracy",   f"{model_info['r2_score']*100:.1f}%")
        st.metric("Avg Error",  f"${model_info['mae']:.0f}")

    st.markdown("---")
    st.markdown("### 👨‍🎓 Project By")
    st.markdown("**Sanjay Thakor**")
    st.markdown("Roll No: 220390107031")
    st.markdown("Course: ML Internship")
    st.markdown("Guide: Prof. Akshay Kansara")
    st.markdown("---")
    st.markdown("*Made with ❤️ using Python*")

# ── HOME ──────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏥 Insurance Premium Predictor</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h2>📊 Data-Driven</h2><p>1,338 real insurance records</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h2>🎯 Accurate</h2><p>87%+ prediction accuracy</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h2>⚡ Fast</h2><p>Instant predictions</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 Features")
        st.markdown("- Real-time insurance charge prediction\n- Interactive data visualizations\n- 6 ML models compared\n- Feature importance analysis\n- Personalized health recommendations")
    with c2:
        st.markdown("### 🚀 How It Works")
        st.markdown("1. **Input** your details (age, BMI, smoking status, etc.)\n2. **AI Model** analyzes your information\n3. **Get** instant insurance charge prediction\n4. **View** personalized tips to reduce premium")
    st.markdown("---")
    st.info("💡 Use the sidebar to navigate to different sections!")

# ── DASHBOARD ─────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Data Insights Dashboard</div>', unsafe_allow_html=True)
    if df is None:
        st.error("Dataset not found!"); st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Age",       f"{df['age'].mean():.0f} years")
    c3.metric("Avg BMI",       f"{df['bmi'].mean():.1f}")
    c4.metric("Avg Premium",   f"${df['charges'].mean():,.0f}")
    st.markdown("---")

    with st.expander("🔧 Filter Data", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        smoker_filter = fc1.multiselect("Smoker",  ["yes","no"],         default=["yes","no"])
        sex_filter    = fc2.multiselect("Gender",  ["male","female"],    default=["male","female"])
        age_range     = fc3.slider("Age Range", int(df['age'].min()), int(df['age'].max()),
                                   (int(df['age'].min()), int(df['age'].max())))

    dff = df[df['smoker'].isin(smoker_filter) & df['sex'].isin(sex_filter) &
             df['age'].between(age_range[0], age_range[1])]
    st.caption(f"Showing {len(dff):,} records after filter")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x='charges', nbins=30, title='💰 Premium Distribution',
                           color_discrete_sequence=['#667eea'], labels={'charges':'Annual Premium ($)'})
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(dff, x='age', y='charges', color='smoker', title='📅 Age vs Premium',
                         color_discrete_map={'yes':'#f5576c','no':'#667eea'}, opacity=0.5,
                         labels={'age':'Age','charges':'Premium ($)'})
        fig.update_layout(height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        smoker_avg = dff.groupby('smoker')['charges'].mean().reset_index()
        fig = px.bar(smoker_avg, x='smoker', y='charges', title='🚬 Avg Premium by Smoking',
                     color='smoker', color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(dff, x='bmi', y='charges', color='smoker', title='⚖️ BMI vs Premium',
                         color_discrete_map={'yes':'#f5576c','no':'#667eea'}, opacity=0.5,
                         labels={'bmi':'BMI','charges':'Premium ($)'})
        fig.update_layout(height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        region_avg = dff.groupby('region')['charges'].mean().reset_index()
        fig = px.bar(region_avg, x='region', y='charges', title='🗺️ Avg Premium by Region',
                     color='charges', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=350, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        children_avg = dff.groupby('children')['charges'].mean().reset_index()
        fig = px.bar(children_avg, x='children', y='charges', title='👶 Avg Premium by Children',
                     color_discrete_sequence=['#764ba2'], text_auto='$.0f')
        fig.update_layout(showlegend=False, height=350, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# ── PREDICT ───────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.markdown('<div class="main-header">🔮 Premium Predictor</div>', unsafe_allow_html=True)
    if model is None:
        st.error("⚠️ Model not found!"); st.stop()

    st.markdown("### Enter Your Details:")
    c1, c2 = st.columns(2)
    with c1:
        age      = st.slider("🎂 Age", 18, 100, 30)
        bmi      = st.slider("⚖️ BMI", 10.0, 60.0, 25.0, 0.1)
        children = st.selectbox("👶 Number of Children", [0,1,2,3,4,5])
    with c2:
        sex    = st.selectbox("👤 Gender", ["male","female"])
        smoker = st.selectbox("🚬 Do you smoke?", ["no","yes"])
        region = st.selectbox("🗺️ Region", ["northeast","northwest","southeast","southwest"])

    if st.button("🔮 Predict My Premium", type="primary"):
        sex_enc    = encoders['sex'].transform([sex])[0]
        smoker_enc = encoders['smoker'].transform([smoker])[0]
        region_enc = encoders['region'].transform([region])[0]

        features = pd.DataFrame([[age, sex_enc, bmi, children, smoker_enc, region_enc,
                                   age*bmi, smoker_enc*age, smoker_enc*bmi]],
                                 columns=['age','sex_encoded','bmi','children','smoker_encoded',
                                          'region_encoded','age_bmi','smoker_age','smoker_bmi'])
        prediction = model.predict(features)[0]

        st.markdown(f'<div class="prediction-box">💰 Predicted Annual Premium: ${prediction:,.2f}</div>',
                    unsafe_allow_html=True)
        st.markdown("### 💡 Tips to Lower Your Premium:")
        c1, c2 = st.columns(2)
        with c1:
            st.warning("🚭 Quitting smoking could save $15,000+ annually!") if smoker == 'yes' else st.success("✅ Non-smoker — great savings!")
            st.info("🏃 Reducing BMI below 30 can lower premium.") if bmi > 30 else st.success("✅ Healthy BMI!")
        with c2:
            if age > 50: st.info("🏥 Regular checkups help manage costs.")
            if children >= 3: st.info("👨‍👩‍👧 More dependents slightly increase premium.")
            if smoker == 'no' and bmi <= 25: st.success("🌟 Lowest risk category!")

# ── ANALYSIS ──────────────────────────────────────────────────
elif page == "📈 Analysis":
    st.markdown('<div class="main-header">📈 Deep Data Analysis</div>', unsafe_allow_html=True)
    if df is None:
        st.error("Dataset not found!"); st.stop()

    # All heavy computation cached
    corr, smoker_corr, s_avg, ns_avg, age_grp, bmi_grp = get_analysis_data(df)

    st.markdown("### 🔍 Feature Correlation Heatmap")
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r',
                    title=f'Correlation Matrix — smoker strongest link to charges ({smoker_corr})')
    fig.update_layout(height=420, paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.markdown("### 🚬 Smoking Impact — The #1 Cost Driver")
    diff = s_avg - ns_avg
    pct  = (s_avg / ns_avg - 1) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Smokers Avg",     f"${s_avg:,.0f}")
    c2.metric("Non-Smokers Avg", f"${ns_avg:,.0f}")
    c3.metric("Extra Cost",      f"${diff:,.0f}", f"+{pct:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x='smoker', y='charges', title='Premium Spread by Smoking',
                     color='smoker', color_discrete_map={'yes':'#f5576c','no':'#667eea'})
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(df, x='smoker', y='charges', title='Distribution Shape by Smoking',
                        color='smoker', color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                        box=True, points=False)
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📅 Age & BMI Effect on Premiums")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(age_grp, x='age_group', y='charges', title='Avg Premium by Age Group',
                     color='charges', color_continuous_scale='Blues', text_auto='$.0f')
        fig.update_layout(showlegend=False, height=360, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(bmi_grp, x='bmi_cat', y='charges', title='Avg Premium by BMI Category',
                     color='charges', color_continuous_scale='Oranges', text_auto='$.0f')
        fig.update_layout(showlegend=False, height=360, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Feature Importance Summary")
    st.dataframe(pd.DataFrame({
        'Feature'     : ['🚬 Smoker','📅 Age','⚖️ BMI','👶 Children','👤 Gender','🗺️ Region'],
        'Correlation' : [0.79, 0.30, 0.20, 0.07, 0.06, 0.01],
        'Impact Level': ['Very High','Moderate','Moderate','Low','Very Low','Negligible'],
        'Insight'     : ['Smokers pay 280% more','Each decade adds ~$3,000',
                         'Obese pay more','Slight increase per child',
                         'Males pay marginally more','Southeast slightly higher']
    }), use_container_width=True, hide_index=True)

# ── MODELS ────────────────────────────────────────────────────
elif page == "🤖 Models":
    st.markdown('<div class="main-header">🤖 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("### 📊 All 6 Models I Tested")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results_df, x='Model', y='Accuracy (%)', title='Model Accuracy Comparison',
                     color='Accuracy (%)', color_continuous_scale='Viridis', text_auto='.2f')
        fig.update_layout(xaxis_tickangle=-30, showlegend=False, height=400, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(results_df, x='Model', y='Error ($)', title='Model Error (Lower = Better)',
                     color='Error ($)', color_continuous_scale='Reds_r', text_auto='$.0f')
        fig.update_layout(xaxis_tickangle=-30, showlegend=False, height=400, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    if model_info:
        st.markdown("### 🏆 Best Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",     model_info['model_name'])
        c2.metric("Accuracy",  f"{model_info['r2_score']*100:.1f}%")
        c3.metric("Avg Error", f"${model_info['mae']:.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.success("**🏆 Why Gradient Boosting Won:**\n- Highest R² score: 87.67%\n- Handles non-linear relationships\n- Robust to outliers\n- Sequential error correction")
    with c2:
        st.info("**📈 Key Patterns:**\n- Ensemble methods performed best\n- Linear models surprisingly competitive\n- Decision Tree overfits badly (72%)\n- Feature engineering boosted all models")

    st.markdown("### 🔍 My Model Selection Process")
    st.markdown("1. **Started** with Linear Regression as baseline\n2. **Tried** Ridge & Lasso to prevent overfitting\n3. **Tested** tree-based models (Decision Tree, Random Forest, Gradient Boosting)\n4. **Selected** Gradient Boosting — best accuracy at 87.67% with $2,425 avg error")

# ── ABOUT ─────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, grad in [
        (c1, "📊", "1,338",   "Records Analyzed",    "linear-gradient(135deg,#667eea,#764ba2)"),
        (c2, "🎯", "87.67%",  "Model Accuracy",       "linear-gradient(135deg,#11998e,#38ef7d)"),
        (c3, "🤖", "6",       "ML Models Tested",     "linear-gradient(135deg,#f093fb,#f5576c)"),
        (c4, "💰", "$2,425",  "Avg Prediction Error", "linear-gradient(135deg,#fc4a1a,#f7b733)"),
    ]:
        col.markdown(f"<div style='background:{grad};border-radius:12px;padding:1.2rem;text-align:center;color:white;'>"
                     f"<div style='font-size:2rem;'>{icon}</div>"
                     f"<div style='font-size:1.5rem;font-weight:800;'>{val}</div>"
                     f"<div style='font-size:0.85rem;opacity:0.85;'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div style='background:white;border-radius:14px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:1rem;'>
            <h3 style='color:#1f77b4;margin-top:0;'>🎯 Project Overview</h3>
            <p style='color:#444;line-height:1.7;'>SmartPolicy is an end-to-end ML project predicting annual insurance premium charges based on personal health and demographic information. Built to make insurance pricing transparent and accessible.</p>
        </div>
        <div style='background:white;border-radius:14px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:1rem;'>
            <h3 style='color:#1f77b4;margin-top:0;'>🛠️ Tech Stack</h3>
            <table style='width:100%;border-collapse:collapse;color:#444;'>
                <tr style='background:#f8f9fa;'><td style='padding:8px 12px;font-weight:600;'>Python</td><td style='padding:8px 12px;'>Core language</td></tr>
                <tr><td style='padding:8px 12px;font-weight:600;'>Pandas & NumPy</td><td style='padding:8px 12px;'>Data processing</td></tr>
                <tr style='background:#f8f9fa;'><td style='padding:8px 12px;font-weight:600;'>Scikit-learn</td><td style='padding:8px 12px;'>ML algorithms</td></tr>
                <tr><td style='padding:8px 12px;font-weight:600;'>Streamlit</td><td style='padding:8px 12px;'>Web application</td></tr>
                <tr style='background:#f8f9fa;'><td style='padding:8px 12px;font-weight:600;'>Plotly</td><td style='padding:8px 12px;'>Interactive charts</td></tr>
            </table>
        </div>
        <div style='background:white;border-radius:14px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,0.08);'>
            <h3 style='color:#1f77b4;margin-top:0;'>📊 Dataset</h3>
            <p style='color:#444;margin:0;'>• <b>1,338 records</b> from real insurance customers<br>• <b>6 features:</b> Age, Sex, BMI, Children, Smoker, Region<br>• <b>Target:</b> Annual charges in USD ($1,121–$63,770)<br>• <b>Source:</b> Kaggle Insurance Dataset</p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:14px;padding:1.5rem;color:white;margin-bottom:1rem;'>
            <h3 style='margin-top:0;color:white;'>👨‍🎓 Student Info</h3>
            <p style='margin:6px 0;'>👤 <b>Sanjay Thakor</b></p>
            <p style='margin:6px 0;'>🎓 Roll No: 220390107031</p>
            <p style='margin:6px 0;'>📚 Course: ML Internship</p>
            <p style='margin:6px 0;'>👨‍🏫 Guide: Prof. Akshay Kansara</p>
            <p style='margin:6px 0;'>📅 Year: 2026</p>
        </div>
        <div style='background:white;border-radius:14px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:1rem;'>
            <h3 style='color:#1f77b4;margin-top:0;'>🔍 Key Discoveries</h3>
            <p style='color:#444;margin:6px 0;'>🚬 Smokers pay <b>280% more</b></p>
            <p style='color:#444;margin:6px 0;'>📅 Age increases cost gradually</p>
            <p style='color:#444;margin:6px 0;'>⚖️ Higher BMI = higher premium</p>
            <p style='color:#444;margin:6px 0;'>🤖 Gradient Boosting = best model</p>
        </div>
        <div style='background:linear-gradient(135deg,#11998e,#38ef7d);border-radius:14px;padding:1.2rem;color:white;text-align:center;'>
            <div style='font-size:1rem;font-weight:600;'>💡 ML can solve real business problems!</div>
        </div>""", unsafe_allow_html=True)
