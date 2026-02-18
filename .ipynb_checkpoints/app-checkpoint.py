import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from gtts import gTTS

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Burnout Intelligence System", layout="wide")

# -------------------------
# PREMIUM STYLING
# -------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
}
.big-title {
    font-size:42px !important;
    font-weight:800;
    color:white;
}
.subtitle {
    font-size:18px;
    color:#DDDDDD;
}
.card {
    background-color: rgba(255,255,255,0.05);
    padding:25px;
    border-radius:15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODELS
# -------------------------
classifier = joblib.load("burnout_classifier.pkl")
regressor = joblib.load("performance_regressor.pkl")
scaler_class = joblib.load("scaler_classification.pkl")
scaler_reg = joblib.load("scaler_regression.pkl")

# -------------------------
# HEADER
# -------------------------
st.markdown('<p class="big-title">🧠 AI Cognitive Burnout Intelligence System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Burnout Risk Prediction with Explainable AI</p>', unsafe_allow_html=True)
st.divider()

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("📥 Student Lifestyle Input")

study = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
sleep = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0)
screen = st.sidebar.slider("Screen Time", 0.0, 12.0, 5.0)
exercise = st.sidebar.slider("Exercise Hours", 0.0, 4.0, 1.0)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
previous = st.sidebar.slider("Previous Marks", 35.0, 100.0, 70.0)

# -------------------------
# RUN ANALYSIS BUTTON
# -------------------------
if st.sidebar.button("🚀 Run AI Analysis"):

    input_data = np.array([[study, sleep, screen, exercise, stress, previous]])

    input_scaled_c = scaler_class.transform(input_data)
    input_scaled_r = scaler_reg.transform(input_data)

    burnout_pred = classifier.predict(input_scaled_c)[0]
    burnout_prob = classifier.predict_proba(input_scaled_c)[0]
    predicted_marks = regressor.predict(input_scaled_r)[0]

    class_labels = ["High", "Low", "Medium"]
    burnout_label = class_labels[burnout_pred]
    risk_score = burnout_prob[0] * 100

    # STORE IN SESSION STATE
    st.session_state["burnout_label"] = burnout_label
    st.session_state["predicted_marks"] = round(predicted_marks, 2)
    st.session_state["burnout_prob"] = burnout_prob
    st.session_state["risk_score"] = risk_score

# -------------------------
# DISPLAY RESULTS IF AVAILABLE
# -------------------------
if "burnout_label" in st.session_state:

    burnout_label = st.session_state["burnout_label"]
    predicted_marks = st.session_state["predicted_marks"]
    burnout_prob = st.session_state["burnout_prob"]
    risk_score = st.session_state["risk_score"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔥 Burnout Risk Level")
        st.write(f"### {burnout_label}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Predicted Final Marks")
        st.write(f"### {predicted_marks}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # RISK GAUGE
    # -------------------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Burnout Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # -------------------------
    # PROBABILITY BAR CHART
    # -------------------------
    prob_df = pd.DataFrame({
        "Burnout Level": ["High", "Low", "Medium"],
        "Probability": burnout_prob
    })

    prob_fig = px.bar(prob_df, x="Burnout Level", y="Probability",
                      color="Burnout Level",
                      title="Burnout Probability Distribution")

    st.plotly_chart(prob_fig, use_container_width=True)

    # -------------------------
    # LIFESTYLE RADAR
    # -------------------------
    radar_fig = go.Figure()

    radar_fig.add_trace(go.Scatterpolar(
        r=[study, sleep, screen, exercise, stress],
        theta=["Study", "Sleep", "Screen", "Exercise", "Stress"],
        fill='toself'
    ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title="Lifestyle Pattern Radar"
    )

    st.plotly_chart(radar_fig, use_container_width=True)

    st.divider()

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
    st.subheader("💡 AI Recommendations")

    recommendations = []

    if sleep < 6:
        recommendations.append("💤 Increase sleep to 7–8 hours.")
    if stress > 7:
        recommendations.append("🧘 Practice stress management techniques.")
    if screen > 7:
        recommendations.append("📵 Reduce screen exposure.")
    if exercise < 0.5:
        recommendations.append("🏃 Include daily physical activity.")
    if burnout_label == "High":
        recommendations.append("📚 Consider academic recovery planning.")

    if recommendations:
        for rec in recommendations:
            st.success(rec)
    else:
        st.success("✔ Lifestyle appears balanced.")

    st.divider()

    # -------------------------
    # VOICE BUTTON
    # -------------------------
    if st.button("🔊 Speak AI Summary"):

        summary_text = f"""
        Burnout level is {burnout_label}.
        Predicted final marks are {predicted_marks}.
        Please follow the recommendations shown on the screen.
        """

        tts = gTTS(summary_text)
        tts.save("voice.mp3")

        with open("voice.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()

        st.audio(audio_bytes, format="audio/mp3")

else:
    st.info("👈 Please enter inputs and click 'Run AI Analysis' to see results.")

st.markdown("---")
st.markdown("Built with Machine Learning + Explainable AI + Interactive Visualization")
