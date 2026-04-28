import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

st.title("🏥 Healthcare Dashboard")
st.write("Governance-Ready Analytics Framework")

np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

st.metric("Model Accuracy", f"{acc:.2%}")
st.success("Dashboard deployed successfully!")
