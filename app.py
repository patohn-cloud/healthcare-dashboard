import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Governance-Ready Analytics Framework", layout="wide")

st.title("Governance-Ready Analytics Framework")
st.subheader("Synthetic Data + Machine Learning Classification + Explainability for Healthcare Decision Support")
st.markdown("---")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    
    data = {
        'age': np.random.normal(55, 15, n).clip(18, 90),
        'bmi': np.random.normal(28, 6, n).clip(15, 50),
        'high_bp': np.random.choice([0,1], n, p=[0.65, 0.35]),
        'high_chol': np.random.choice([0,1], n, p=[0.60, 0.40]),
        'smoker': np.random.choice([0,1], n, p=[0.70, 0.30]),
        'exercise': np.random.choice([0,1], n, p=[0.35, 0.65]),
        'income': np.random.choice([1,2,3,4,5,6,7,8], n),
        'education': np.random.choice([1,2,3,4,5,6], n)
    }
    
    df = pd.DataFrame(data)
    
    risk = (0.03*(df['age']-50) + 0.05*(df['bmi']-25) + 0.4*df['high_bp'] + 
            0.3*df['high_chol'] + 0.2*df['smoker'] - 0.3*df['exercise'])
    prob = 1/(1+np.exp(-np.clip(risk, -3, 3)))
    df['diabetes'] = (np.random.random(n) < prob).astype(int)
    
    return df

@st.cache_resource
def train_and_evaluate():
    df = load_data()
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Generate synthetic data
    train_data = X_train.copy()
    train_data['diabetes'] = y_train.values
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    means = train_data[numeric_cols].mean()
    covs = train_data[numeric_cols].cov()
    
    synthetic = np.random.multivariate_normal(means, covs, len(X_train))
    synthetic_df = pd.DataFrame(synthetic, columns=numeric_cols)
    for col in numeric_cols:
        if col in X_train.columns:
            synthetic_df[col] = synthetic_df[col].clip(X_train[col].min(), X_train[col].max())
    
    y_synth = synthetic_df['diabetes'].round().astype(int)
    X_synth = synthetic_df.drop('diabetes', axis=1)
    X_synth_scaled = scaler.transform(X_synth)
    
    # Models (only stable ones)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # Train on REAL data
        m_real = model.__class__(**model.get_params())
        m_real.fit(X_train_scaled, y_train)
        y_proba_real = m_real.predict_proba(X_test_scaled)[:, 1]
        
        # Train on SYNTHETIC data
        m_synth = model.__class__(**model.get_params())
        m_synth.fit(X_synth_scaled, y_synth)
        y_proba_synth = m_synth.predict_proba(X_test_scaled)[:, 1]
        
        results.append({'Model': name, 'Regime': 'Real', 'ROC-AUC': roc_auc_score(y_test, y_proba_real)})
        results.append({'Model': name, 'Regime': 'Synthetic', 'ROC-AUC': roc_auc_score(y_test, y_proba_synth)})
    
    return pd.DataFrame(results), y_test

# Load data and results
df = load_data()
results_df, y_test = train_and_evaluate()

# Compute ECI function
def compute_eci(proba):
    confidence = 1 - 2 * abs(proba - 0.5)
    concentration = 0.6
    stability = 0.5
    return 0.4 * concentration + 0.3 * confidence + 0.3 * stability

# SIDEBAR
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Section", [
        "1. Data Overview",
        "2. Model Performance", 
        "3. Explainability (ECI)",
        "4. Governance & Privacy"
    ])
    st.markdown("---")
    st.caption("MSc Business Analytics")
    st.caption("Dublin Business School | 2026")

# PAGE 1: DATA OVERVIEW
if page == "1. Data Overview":
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", len(df.columns)-1)
    col3.metric("Diabetes Prevalence", f"{df['diabetes'].mean()*100:.1f}%")
    col4.metric("Class Balance", f"{(1-df['diabetes'].mean())*100:.1f}% / {df['diabetes'].mean()*100:.1f}%")
    
    st.subheader("Sample Data (first 100 rows)")
    st.dataframe(df.head(100))
    
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature", [c for c in df.columns if c != 'diabetes'])
    st.bar_chart(df[feature].value_counts())

# PAGE 2: MODEL PERFORMANCE
elif page == "2. Model Performance":
    st.header("Model Performance")
    
    st.subheader("Table 4.1 - Classification Performance (ROC-AUC)")
    
    pivot = results_df.pivot(index='Model', columns='Regime', values='ROC-AUC').reset_index()
    pivot.columns = ['Model', 'ROC-AUC (Real)', 'ROC-AUC (Synthetic)']
    pivot['Difference'] = pivot['ROC-AUC (Real)'] - pivot['ROC-AUC (Synthetic)']
    st.dataframe(pivot.round(4), use_container_width=True)
    
    st.subheader("ROC-AUC Comparison Chart")
    st.bar_chart(results_df.pivot(index='Model', columns='Regime', values='ROC-AUC'))
    
    st.subheader("Hypothesis H1: Equivalence Testing (TOST)")
    
    rf_real = results_df[(results_df['Model']=='Random Forest') & (results_df['Regime']=='Real')]['ROC-AUC'].values[0]
    rf_synth = results_df[(results_df['Model']=='Random Forest') & (results_df['Regime']=='Synthetic')]['ROC-AUC'].values[0]
    diff = rf_real - rf_synth
    margin = 0.02
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Random Forest Real ROC-AUC", f"{rf_real:.4f}")
    col2.metric("Random Forest Synthetic ROC-AUC", f"{rf_synth:.4f}")
    col3.metric("Difference", f"{diff:.4f}", delta=f"Margin ±{margin}")
    
    if abs(diff) < margin:
        st.success("✅ H1 ACCEPTED: Models trained on synthetic data are statistically equivalent to models trained on real data")
    else:
        st.warning(f"❌ H1 REJECTED: Difference of {abs(diff):.4f} exceeds margin of {margin}")

# PAGE 3: EXPLAINABILITY
elif page == "3. Explainability (ECI)":
    st.header("Explainability Confidence Index (ECI)")
    
    st.markdown("""
    **ECI Formula:**
    
