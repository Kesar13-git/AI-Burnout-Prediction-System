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

st.set_page_config(page_title="AI Burnout System", layout="wide")

# ------------------ PREMIUM STYLING ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00e6ff;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("AI_Cognitive_Burnout_Dataset_8000_Rows.csv")

df = load_data()

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

le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class_encoded, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_scaled, y_train_c)

regressor = RandomForestRegressor(
    n_estimators=120,
    max_depth=12,
    random_state=42
)
regressor.fit(X_train_r, y_train_r)

# ------------------ UI ------------------
st.title("🧠 AI Cognitive Burnout Intelligence Dashboard")
st.markdown("##### Advanced ML-powered cognitive analytics system")

st.sidebar.header("📥 Student Input")

study = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
sleep = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0)
screen = st.sidebar.slider("Screen Time", 0.0, 12.0, 5.0)
exercise = st.sidebar.slider("Exercise Hours", 0.0, 4.0, 1.0)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)
previous = st.sidebar.slider("Previous Marks", 35.0, 100.0, 70.0)

if st.sidebar.button("🚀 Run AI Analysis"):

    input_data = np.array([[study, sleep, screen, exercise, stress, previous]])
    input_scaled = scaler.transform(input_data)

    burnout_pred = classifier.predict(input_scaled)[0]
    burnout_prob = classifier.predict_proba(input_scaled)[0]
    predicted_marks = regressor.predict(input_data)[0]

    burnout_label = le.inverse_transform([burnout_pred])[0]
    risk_score = max(burnout_prob) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Burnout Level")
        st.success(burnout_label)

    with col2:
        st.markdown("### 📊 Predicted Final Marks")
        st.info(round(predicted_marks, 2))

    st.divider()

    # -------- Gauge Chart --------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Burnout Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00e6ff"},
            'steps': [
                {'range': [0, 40], 'color': "#00ff99"},
                {'range': [40, 70], 'color': "#ffcc00"},
                {'range': [70, 100], 'color': "#ff4d4d"}
            ],
        }
    ))
    gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gauge, use_container_width=True)

    # -------- Probability Bar --------
    prob_fig = go.Figure()
    prob_fig.add_trace(go.Bar(
        x=le.classes_,
        y=burnout_prob,
        marker_color=["#ff4d4d", "#00ff99", "#ffaa00"]
    ))
    prob_fig.update_layout(
        title="Burnout Probability Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # -------- Radar Chart --------
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[study, sleep, screen, exercise, stress],
        theta=["Study", "Sleep", "Screen", "Exercise", "Stress"],
        fill='toself',
        line_color="#00e6ff"
    ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True)
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # -------- Recommendations --------
    st.markdown("### 💡 Recommendations")

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

    # -------- Voice Assistant --------
    summary = f"""
    Burnout level is {burnout_label}.
    Predicted final marks are {round(predicted_marks,2)}.
    Burnout risk score is {round(risk_score,1)} percent.
    """

    try:
        tts = gTTS(summary)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.markdown("### 🔊 AI Voice Summary")
        st.audio(audio_bytes)
    except:
        st.info("Voice assistant not available.")
