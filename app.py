import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from gtts import gTTS
import io

st.set_page_config(page_title="AI Cognitive Burnout System", layout="wide")

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AI_Cognitive_Burnout_Dataset_8000_Rows.csv")
    return df

df = load_data()

# --------------------------
# FEATURES
# --------------------------
features = [
    "Study_Hours",
    "Sleep_Hours",
    "Screen_Time_Hours",
    "Exercise_Hours",
    "Stress_Level",
    "Previous_Marks"
]

X = df[features]
y_class = df["Burnout_Level"]
y_reg = df["Final_Predicted_Marks"]

# Encode burnout labels
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

# Train/Test split
X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class_encoded, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Scale classification features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --------------------------
# TRAIN MODELS (Lightweight)
# --------------------------
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_scaled, y_train_c)

regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
regressor.fit(X_train_r, y_train_r)

# --------------------------
# UI
# --------------------------
st.title("🧠 AI Cognitive Burnout Intelligence System")
st.markdown("### Predict Burnout Risk & Academic Performance")

st.sidebar.header("📥 Student Input")

study = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
sleep = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0)
screen = st.sidebar.slider("Screen Time", 0.0, 12.0, 5.0)
exercise = st.sidebar.slider("Exercise Hours", 0.0, 4.0, 1.0)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
previous = st.sidebar.slider("Previous Marks", 35.0, 100.0, 70.0)

if st.sidebar.button("🚀 Run AI Analysis"):

    input_data = np.array([[study, sleep, screen, exercise, stress, previous]])
    input_scaled = scaler.transform(input_data)

    burnout_pred = classifier.predict(input_scaled)[0]
    burnout_prob = classifier.predict_proba(input_scaled)[0]
    predicted_marks = regressor.predict(input_data)[0]

    burnout_label = le.inverse_transform([burnout_pred])[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Burnout Level")
        st.success(burnout_label)

    with col2:
        st.subheader("📊 Predicted Final Marks")
        st.info(round(predicted_marks, 2))

    st.divider()

    # --------------------------
    # GAUGE CHART
    # --------------------------
    risk_score = max(burnout_prob) * 100

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Burnout Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # --------------------------
    # PROBABILITY BAR CHART
    # --------------------------
    st.subheader("📊 Burnout Probability Distribution")

    prob_fig = go.Figure(data=[
        go.Bar(
            x=le.classes_,
            y=burnout_prob
        )
    ])

    st.plotly_chart(prob_fig, use_container_width=True)

    # --------------------------
    # RADAR CHART
    # --------------------------
    st.subheader("🧭 Lifestyle Radar Analysis")

    radar_fig = go.Figure()

    radar_fig.add_trace(go.Scatterpolar(
        r=[study, sleep, screen, exercise, stress],
        theta=["Study", "Sleep", "Screen", "Exercise", "Stress"],
        fill='toself'
    ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )

    st.plotly_chart(radar_fig, use_container_width=True)

    # --------------------------
    # FEATURE IMPORTANCE
    # --------------------------
    st.subheader("📈 Feature Importance (Regression Model)")

    importance = regressor.feature_importances_

    importance_fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=importance
        )
    ])

    st.plotly_chart(importance_fig, use_container_width=True)

    # --------------------------
    # RECOMMENDATIONS
    # --------------------------
    st.subheader("💡 Recommendations")

    recommendations = []

    if sleep < 6:
        recommendations.append("Increase sleep to 7–8 hours.")
    if stress > 7:
        recommendations.append("Practice stress reduction techniques.")
    if screen > 7:
        recommendations.append("Reduce screen exposure.")
    if exercise < 0.5:
        recommendations.append("Include daily physical activity.")

    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("Lifestyle appears balanced.")

    # --------------------------
    # VOICE ASSISTANT
    # --------------------------
    summary = f"""
    Burnout level is {burnout_label}.
    Predicted final marks are {round(predicted_marks,2)}.
    Burnout risk score is {round(risk_score,1)} percent.
    """

    try:
        tts = gTTS(summary)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.subheader("🔊 AI Voice Summary")
        st.audio(audio_bytes)
    except:
        st.info("Voice assistant not available in cloud environment.")
