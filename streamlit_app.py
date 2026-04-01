import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-1px);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: white !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #38ef7d !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #aaa !important; }

    /* Nav card buttons */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: rgba(255,255,255,0.07) !important;
        border: 2px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: white !important;
        text-align: left !important;
        padding: 12px 16px !important;
        margin: 3px 0 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: background 0.25s ease, border 0.25s ease,
                    box-shadow 0.25s ease, letter-spacing 0.25s ease !important;
        letter-spacing: 0px !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, rgba(102,126,234,0.7), rgba(118,75,162,0.7)) !important;
        border: 2px solid #667eea !important;
        box-shadow: 0 0 14px rgba(102,126,234,0.55),
                    inset 0 0 8px rgba(255,255,255,0.05) !important;
        letter-spacing: 0.5px !important;
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus,
    [data-testid="stSidebar"] [data-testid="stButton"] button:active {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: 2px solid #a78bfa !important;
        box-shadow: 0 0 20px rgba(102,126,234,0.7) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        model      = joblib.load(os.path.join(base, 'best_model.pkl'))
        encoders   = joblib.load(os.path.join(base, 'encoders.pkl'))
        model_info = joblib.load(os.path.join(base, 'model_info.pkl'))
        return model, encoders, model_info
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        import os
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, 'insurance (1).csv')
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"⚠️ Dataset not found: {e}")
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

model, encoders, model_info = load_model()
df = load_data()
results_df = get_model_results()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Insurance Predictor")
    st.markdown("*Student ML Project*")
    st.markdown("---")

    # Card-style navigation using session state
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"

    nav_items = [
        ("🏠", "Home"),
        ("📊", "Dashboard"),
        ("🔮", "Predict"),
        ("📈", "Analysis"),
        ("🤖", "Models"),
        ("ℹ️", "About"),
    ]

    for icon, label in nav_items:
        key = f"{icon} {label}"
        is_active = st.session_state.page == key
        if st.sidebar.button(
            f"{icon}  {label}",
            key=key,
            use_container_width=True,
        ):
            st.session_state.page = key

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
        st.markdown("""
        - Real-time insurance charge prediction
        - Interactive data visualizations
        - 6 ML models compared
        - Feature importance analysis
        - Personalized health recommendations
        """)
    with c2:
        st.markdown("### 🚀 How It Works")
        st.markdown("""
        1. **Input** your details (age, BMI, smoking status, etc.)
        2. **AI Model** analyzes your information
        3. **Get** instant insurance charge prediction
        4. **View** personalized tips to reduce premium
        """)
    st.markdown("---")
    st.info("💡 Use the sidebar to navigate to different sections!")

# ── DASHBOARD ─────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Data Insights Dashboard</div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found!")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Avg Age",         f"{df['age'].mean():.0f} years")
    c3.metric("Avg BMI",         f"{df['bmi'].mean():.1f}")
    c4.metric("Avg Premium",     f"${df['charges'].mean():,.0f}")

    st.markdown("---")

    # Filter widget for interactivity
    with st.expander("🔧 Filter Data", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        smoker_filter = fc1.multiselect("Smoker", ["yes","no"], default=["yes","no"])
        sex_filter    = fc2.multiselect("Gender", ["male","female"], default=["male","female"])
        age_range     = fc3.slider("Age Range", int(df['age'].min()), int(df['age'].max()),
                                   (int(df['age'].min()), int(df['age'].max())))

    dff = df[
        df['smoker'].isin(smoker_filter) &
        df['sex'].isin(sex_filter) &
        df['age'].between(age_range[0], age_range[1])
    ]
    st.caption(f"Showing {len(dff):,} records after filter")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(dff, x='charges', nbins=30,
                           title='💰 Premium Distribution',
                           color_discrete_sequence=['#667eea'],
                           labels={'charges':'Annual Premium ($)'})
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        smoker_avg = dff.groupby('smoker')['charges'].mean().reset_index()
        fig = px.bar(smoker_avg, x='smoker', y='charges',
                     title='🚬 Avg Premium by Smoking Status',
                     color='smoker',
                     color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                     labels={'smoker':'Smoking Status','charges':'Avg Premium ($)'},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(dff, x='age', y='charges', color='smoker',
                         title='📅 Age vs Premium',
                         color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                         labels={'age':'Age','charges':'Premium ($)','smoker':'Smoker'},
                         opacity=0.6, hover_data=['bmi','children','region'])
        fig.update_layout(height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(dff, x='bmi', y='charges', color='smoker',
                         title='⚖️ BMI vs Premium',
                         color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                         labels={'bmi':'BMI','charges':'Premium ($)','smoker':'Smoker'},
                         opacity=0.6, hover_data=['age','children','region'])
        fig.update_layout(height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        region_avg = dff.groupby('region')['charges'].mean().reset_index()
        fig = px.bar(region_avg, x='region', y='charges',
                     title='🗺️ Avg Premium by Region',
                     color='charges', color_continuous_scale='Blues',
                     labels={'region':'Region','charges':'Avg Premium ($)'})
        fig.update_layout(showlegend=False, height=350, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        children_avg = dff.groupby('children')['charges'].mean().reset_index()
        fig = px.bar(children_avg, x='children', y='charges',
                     title='👶 Avg Premium by No. of Children',
                     color_discrete_sequence=['#764ba2'],
                     labels={'children':'Children','charges':'Avg Premium ($)'},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=350, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# ── PREDICT ───────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.markdown('<div class="main-header">🔮 Premium Predictor</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found! Please run the training notebook first.")
        st.stop()

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
        sex_encoded    = encoders['sex'].transform([sex])[0]
        smoker_encoded = encoders['smoker'].transform([smoker])[0]
        region_encoded = encoders['region'].transform([region])[0]

        age_bmi    = age * bmi
        smoker_age = smoker_encoded * age
        smoker_bmi = smoker_encoded * bmi

        features = pd.DataFrame(
            [[age, sex_encoded, bmi, children, smoker_encoded,
              region_encoded, age_bmi, smoker_age, smoker_bmi]],
            columns=['age','sex_encoded','bmi','children','smoker_encoded',
                     'region_encoded','age_bmi','smoker_age','smoker_bmi']
        )
        prediction = model.predict(features)[0]

        st.markdown(f'<div class="prediction-box">💰 Predicted Annual Premium: ${prediction:,.2f}</div>',
                    unsafe_allow_html=True)

        st.markdown("### 💡 Tips to Lower Your Premium:")
        c1, c2 = st.columns(2)
        with c1:
            if smoker == 'yes':
                st.warning("🚭 Quitting smoking could save you $15,000+ annually!")
            else:
                st.success("✅ Non-smoker — you're saving a lot already!")
            if bmi > 30:
                st.info("🏃 Reducing BMI below 30 can lower your premium.")
            elif bmi <= 25:
                st.success("✅ Healthy BMI — great for lower costs!")
        with c2:
            if age > 50:
                st.info("🏥 Regular health checkups help manage costs.")
            if children >= 3:
                st.info("👨‍👩‍👧 More dependents slightly increase premium.")
            if smoker == 'no' and bmi <= 25:
                st.success("🌟 You're in the lowest risk category!")

# ── ANALYSIS ──────────────────────────────────────────────────
elif page == "📈 Analysis":
    st.markdown('<div class="main-header">📈 Deep Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Understanding WHY premiums vary — statistical insights & feature relationships*")

    if df is None:
        st.error("Dataset not found!")
        st.stop()

    # 1. Correlation heatmap
    st.markdown("### 🔍 Feature Correlation Heatmap")
    st.caption("Shows how strongly each feature is related to insurance charges")
    df_enc = df.copy()
    df_enc['sex']    = df_enc['sex'].map({'male':1,'female':0})
    df_enc['smoker'] = df_enc['smoker'].map({'yes':1,'no':0})
    df_enc['region'] = df_enc['region'].astype('category').cat.codes
    corr = df_enc.corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix — smoker has strongest link to charges (0.79)')
    fig.update_layout(height=420, paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. Smoking deep dive
    st.markdown("### 🚬 Smoking Impact — The #1 Cost Driver")
    s_avg  = df[df['smoker']=='yes']['charges'].mean()
    ns_avg = df[df['smoker']=='no']['charges'].mean()
    diff   = s_avg - ns_avg
    pct    = (s_avg / ns_avg - 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Smokers Avg Premium",     f"${s_avg:,.0f}")
    c2.metric("Non-Smokers Avg Premium", f"${ns_avg:,.0f}")
    c3.metric("Extra Cost for Smoking",  f"${diff:,.0f}", f"+{pct:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        # Box plot — distribution spread
        fig = px.box(df, x='smoker', y='charges',
                     title='Premium Spread: Smoker vs Non-Smoker',
                     color='smoker',
                     color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                     labels={'smoker':'Smoking Status','charges':'Premium ($)'})
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Violin plot — full distribution shape
        fig = px.violin(df, x='smoker', y='charges',
                        title='Premium Distribution Shape by Smoking',
                        color='smoker',
                        color_discrete_map={'yes':'#f5576c','no':'#667eea'},
                        box=True, points=False,
                        labels={'smoker':'Smoking Status','charges':'Premium ($)'})
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Age & BMI effect
    st.markdown("### 📅 Age & BMI Effect on Premiums")
    c1, c2 = st.columns(2)
    with c1:
        # Age group analysis
        df['age_group'] = pd.cut(df['age'], bins=[17,25,35,45,55,65],
                                  labels=['18-25','26-35','36-45','46-55','56-64'])
        age_grp = df.groupby('age_group', observed=True)['charges'].mean().reset_index()
        fig = px.bar(age_grp, x='age_group', y='charges',
                     title='Average Premium by Age Group',
                     color='charges', color_continuous_scale='Blues',
                     labels={'age_group':'Age Group','charges':'Avg Premium ($)'},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=360, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # BMI category analysis
        df['bmi_cat'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100],
                                labels=['Underweight','Normal','Overweight','Obese'])
        bmi_grp = df.groupby('bmi_cat', observed=True)['charges'].mean().reset_index()
        fig = px.bar(bmi_grp, x='bmi_cat', y='charges',
                     title='Average Premium by BMI Category',
                     color='charges', color_continuous_scale='Oranges',
                     labels={'bmi_cat':'BMI Category','charges':'Avg Premium ($)'},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=360, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 4. Summary table
    st.markdown("### 📊 Feature Importance Summary")
    summary = pd.DataFrame({
        'Feature'     : ['🚬 Smoker','📅 Age','⚖️ BMI','👶 Children','👤 Gender','🗺️ Region'],
        'Correlation' : [0.79, 0.30, 0.20, 0.07, 0.06, 0.01],
        'Impact Level': ['Very High','Moderate','Moderate','Low','Very Low','Negligible'],
        'Insight'     : [
            'Smokers pay 280% more',
            'Each decade adds ~$3,000',
            'Obese pay more than normal BMI',
            'Slight increase per child',
            'Males pay marginally more',
            'Southeast slightly higher'
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # cleanup temp columns
    df.drop(columns=['age_group','bmi_cat'], inplace=True, errors='ignore')

# ── MODELS ────────────────────────────────────────────────────
elif page == "🤖 Models":
    st.markdown('<div class="main-header">🤖 Model Comparison</div>', unsafe_allow_html=True)

    st.markdown("### 📊 All 6 Models I Tested")
    st.markdown("I compared 6 different algorithms to find the best one for predicting insurance premiums.")

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results_df, x='Model', y='Accuracy (%)',
                     title='Model Accuracy Comparison',
                     color='Accuracy (%)', color_continuous_scale='Viridis',
                     text_auto='.2f')
        fig.update_layout(xaxis_tickangle=-30, showlegend=False,
                          height=400, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(results_df, x='Model', y='Error ($)',
                     title='Model Error Comparison (Lower = Better)',
                     color='Error ($)', color_continuous_scale='Reds_r',
                     text_auto='$.0f')
        fig.update_layout(xaxis_tickangle=-30, showlegend=False,
                          height=400, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    if model_info:
        st.markdown("### 🏆 Best Model Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",     model_info['model_name'])
        c2.metric("Accuracy",  f"{model_info['r2_score']*100:.1f}%")
        c3.metric("Avg Error", f"${model_info['mae']:.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.success("""
        **🏆 Why Gradient Boosting Won:**
        - Highest R² score: 87.67%
        - Handles complex non-linear relationships
        - Robust to outliers in data
        - Sequential error correction
        """)
    with c2:
        st.info("""
        **📈 Key Patterns:**
        - Ensemble methods (GB, RF) performed best
        - Linear models were surprisingly competitive
        - Single Decision Tree overfits badly (72%)
        - Feature engineering boosted all models
        """)

    st.markdown("### 🔍 My Model Selection Process")
    st.markdown("""
    1. **Started** with Linear Regression as baseline
    2. **Tried** Ridge & Lasso to prevent overfitting
    3. **Tested** tree-based models (Decision Tree, Random Forest, Gradient Boosting)
    4. **Selected** Gradient Boosting — best accuracy at 87.67% with $2,425 avg error
    """)

# ── ABOUT ─────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    # Top banner cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px;padding:1.2rem;text-align:center;color:white;'>
            <div style='font-size:2rem;'>📊</div>
            <div style='font-size:1.5rem;font-weight:800;'>1,338</div>
            <div style='font-size:0.85rem;opacity:0.85;'>Records Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#11998e,#38ef7d);
                    border-radius:12px;padding:1.2rem;text-align:center;color:white;'>
            <div style='font-size:2rem;'>🎯</div>
            <div style='font-size:1.5rem;font-weight:800;'>87.67%</div>
            <div style='font-size:0.85rem;opacity:0.85;'>Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#f093fb,#f5576c);
                    border-radius:12px;padding:1.2rem;text-align:center;color:white;'>
            <div style='font-size:2rem;'>🤖</div>
            <div style='font-size:1.5rem;font-weight:800;'>6</div>
            <div style='font-size:0.85rem;opacity:0.85;'>ML Models Tested</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#fc4a1a,#f7b733);
                    border-radius:12px;padding:1.2rem;text-align:center;color:white;'>
            <div style='font-size:2rem;'>💰</div>
            <div style='font-size:1.5rem;font-weight:800;'>$2,425</div>
            <div style='font-size:0.85rem;opacity:0.85;'>Avg Prediction Error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div style='background:white;border-radius:14px;padding:1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:1rem;'>
            <h3 style='color:#1f77b4;margin-top:0;'>🎯 Project Overview</h3>
            <p style='color:#444;line-height:1.7;'>
            SmartPolicy is an end-to-end machine learning project that predicts annual
            insurance premium charges based on personal health and demographic information.
            I built this to understand how data science can solve real business problems —
            making insurance pricing transparent and accessible to everyone.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:white;border-radius:14px;padding:1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:1rem;'>
            <h3 style='color:#1f77b4;margin-top:0;'>🛠️ Tech Stack</h3>
            <table style='width:100%;border-collapse:collapse;color:#444;'>
                <tr style='background:#f8f9fa;'>
                    <td style='padding:8px 12px;font-weight:600;'>Python 3.12</td>
                    <td style='padding:8px 12px;'>Core programming language</td>
                </tr>
                <tr>
                    <td style='padding:8px 12px;font-weight:600;'>Pandas & NumPy</td>
                    <td style='padding:8px 12px;'>Data manipulation & analysis</td>
                </tr>
                <tr style='background:#f8f9fa;'>
                    <td style='padding:8px 12px;font-weight:600;'>Scikit-learn</td>
                    <td style='padding:8px 12px;'>ML algorithms & preprocessing</td>
                </tr>
                <tr>
                    <td style='padding:8px 12px;font-weight:600;'>Streamlit</td>
                    <td style='padding:8px 12px;'>Interactive web application</td>
                </tr>
                <tr style='background:#f8f9fa;'>
                    <td style='padding:8px 12px;font-weight:600;'>Plotly</td>
                    <td style='padding:8px 12px;'>Interactive visualizations</td>
                </tr>
            </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:white;border-radius:14px;padding:1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.08);'>
            <h3 style='color:#1f77b4;margin-top:0;'>📊 Dataset</h3>
            <p style='color:#444;margin:0;'>
            • <b>1,338 records</b> from real insurance customers<br>
            • <b>6 input features:</b> Age, Sex, BMI, Children, Smoker, Region<br>
            • <b>Target:</b> Annual insurance charges in USD ($1,121 – $63,770)<br>
            • <b>Source:</b> Kaggle Insurance Dataset
            </p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border-radius:14px;padding:1.5rem;color:white;margin-bottom:1rem;'>
            <h3 style='margin-top:0;color:white;'>👨‍🎓 Student Info</h3>
            <p style='margin:6px 0;'>👤 <b>Sanjay Thakor</b></p>
            <p style='margin:6px 0;'>🎓 Roll No: 220390107031</p>
            <p style='margin:6px 0;'>📚 Course: ML Internship</p>
            <p style='margin:6px 0;'>👨‍🏫 Guide: Prof. Akshay Kansara</p>
            <p style='margin:6px 0;'>📅 Year: 2026</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:white;border-radius:14px;padding:1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:1rem;'>
            <h3 style='color:#1f77b4;margin-top:0;'>🔍 Key Discoveries</h3>
            <p style='color:#444;margin:6px 0;'>🚬 Smokers pay <b>280% more</b></p>
            <p style='color:#444;margin:6px 0;'>📅 Age increases cost gradually</p>
            <p style='color:#444;margin:6px 0;'>⚖️ Higher BMI = higher premium</p>
            <p style='color:#444;margin:6px 0;'>🤖 Gradient Boosting = best model</p>
            <p style='color:#444;margin:6px 0;'>🔧 Feature engineering improved accuracy</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:linear-gradient(135deg,#11998e,#38ef7d);
                    border-radius:14px;padding:1.2rem;color:white;text-align:center;'>
            <div style='font-size:1rem;font-weight:600;'>
            💡 This project taught me how ML can solve real business problems!
            </div>
        </div>""", unsafe_allow_html=True)
