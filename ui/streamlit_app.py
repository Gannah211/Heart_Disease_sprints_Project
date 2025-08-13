import streamlit as st 
import pandas as pd 
import json
from pathlib import Path
import joblib 

st.title("Heart Disease Prediction App")

model = joblib.load('models/final_model.pkl')
meta_path = Path("../notebooks/model_meta.json")

if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    THRESHOLD = float(meta.get("threshold", 0.5))
    FEATURE_NAMES = meta.get("features", ['age','sex','cp','trestbps','chol','fbs','restecg',
                                          'thalach','exang','oldpeak','slope','ca','thal'])
else:
    THRESHOLD = 0.5
    FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                     'thalach','exang','oldpeak','slope','ca','thal']

st.caption(f"Decision threshold = {THRESHOLD:.3f}")


age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0,1,2,3,4])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum cholesterol", value=200)
fbs =st.selectbox("Fasing Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Resting ECG Results",[0,1,2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.selectbox("Slope of peak Exercise ST segment",[0,1,2,3])
ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
thal = st.selectbox("Thalassemia",[0,1,2,3,4,5,6,7])


sex_val = 1 if sex == "Male" else 0

def make_row():
    return pd.DataFrame([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]],
                        columns=FEATURE_NAMES)

if st.button("Predict"):
    row = make_row()
    proba = model.predict_proba(row)[:, 1][0]
    pred = int(proba >= THRESHOLD)
    st.write(f"Probability of disease: {proba:.3f}")

    if pred == 0:
        st.success("The model predicts no risk of heart disease.")
    else:
        st.error("The model predicts a risk of heart disease.")