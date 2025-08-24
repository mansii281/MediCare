# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import requests
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for PDF
import matplotlib.cm as cm

# ========================= PAGE CONFIG =========================
st.set_page_config(page_title="🩺 MediTrack", layout="wide", initial_sidebar_state="expanded")

# ========================= LIGHT GRADIENT BACKGROUND =========================
page_bg = """
<style>
body {
    background: linear-gradient(135deg, #F9F9F9, #E0F7FA);
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F9F9F9, #E0F7FA);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    background: rgba(0,0,0,0);
}
.css-1cpxqw2.edgvbvh3 {  /* enlarge number_input / text_area */
    font-size: 18px;
    padding: 8px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ========================= HEADER =========================
st.markdown("""
<div style="text-align:center; padding:20px;
            background: linear-gradient(135deg, #A8E6CF, #DCEDC1);
            border-radius:15px;">
  <h1>🩺 MediTrack</h1>
  <h4>AI-Powered Health Risk Detector & Tracker</h4>
  <p>Enter patient data to get smart risk assessment and insights</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ========================= LOAD MODELS =========================
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_cls.pkl")
    xgb_model = joblib.load("xgb_cls.pkl")
    return rf_model, xgb_model

rf_model, xgb_model = load_models()

# ========================= INPUT FORM =========================
st.subheader("📝 Patient Health Information")
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("👤 Age", 1, 120, 30, step=1)
        cholesterol = st.number_input("🧪 Cholesterol (mg/dL)", 100, 400, 180, step=1)
    with col2:
        bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 22.0, step=0.1)
        glucose = st.number_input("🍬 Glucose (mg/dL)", 70, 300, 100, step=1)
    with col3:
        blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", 80, 200, 120, step=1)
        heart_rate = st.number_input("❤️ Heart Rate (bpm)", 40, 200, 72, step=1)
    doctor_notes = st.text_area("📝 Doctor Notes (Optional)", height=100)
    submitted = st.form_submit_button("🔍 Predict Risk")

# ========================= SIDEBAR =========================
st.sidebar.header("📋 Patient Summary")
patient_summary = {"Age": age,"BMI": bmi,"BloodPressure": blood_pressure,
                   "Cholesterol": cholesterol,"Glucose": glucose,"HeartRate": heart_rate}
for k, v in patient_summary.items():
    st.sidebar.write(f"**{k}:** {v}")

# ========================= PREDICTION =========================
if submitted:
    feature_names = ["Age","BMI","BloodPressure","Cholesterol","Glucose","HeartRate"]
    input_df = pd.DataFrame([[age, bmi, blood_pressure, cholesterol, glucose, heart_rate]],
                            columns=feature_names)

    # Predict probabilities
    rf_pred = rf_model.predict_proba(input_df)[0][1]
    xgb_pred = xgb_model.predict_proba(input_df)[0][1]
    avg_risk = (rf_pred + xgb_pred)/2

    if avg_risk >= 0.5:
        risk_level = "High Risk"
        risk_color = "#FF4C4C"
        risk_emoji = "🚨"
        diagnosis = "Patient is at high risk! Consult a doctor immediately."
        risk_img_url = "https://static.vecteezy.com/system/resources/thumbnails/002/557/001/small/high-risk-concept-on-speedometer-illustration-speedometer-icon-colorful-infographic-gauge-element-vector.jpg"
        card_grad = "linear-gradient(135deg, #FFCDD2, #EF9A9A)"
    else:
        risk_level = "Low Risk"
        risk_color = "#4CAF50"
        risk_emoji = "✅"
        diagnosis = "Patient is at low risk. Maintain healthy lifestyle."
        risk_img_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6SxGNqDrK3fVsJccM6DubO-yQjjwvdA7UkQ&s"
        card_grad = "linear-gradient(135deg, #C8E6C9, #A5D6A7)"

    # ========================= RISK CARD (SMALLER) =========================
    st.markdown("### 📊 Risk Assessment")
    col1, col2 = st.columns([2,1])
    col1.markdown(f"""
    <div style='
        padding:15px; 
        border-radius:15px; 
        background: {card_grad}; 
        text-align:center; 
        box-shadow: 3px 3px 15px rgba(0,0,0,0.15);
    '>
        <h5>🧮 Final Risk Score</h5>
        <h3 style='color:{risk_color}; font-size:2em;'>{avg_risk:.2f}</h3>
        <h5>Risk Level: {risk_level} {risk_emoji}</h5>
    </div>
    """, unsafe_allow_html=True)
    col2.image(risk_img_url, width=120)

    # ========================= PROGRESS BAR (SMALLER) =========================
    st.markdown(f"""
    <div style="background:#E0E0E0; border-radius:15px; width:100%; height:15px;">
      <div style="width:{avg_risk*100}%; background:{risk_color}; height:15px; border-radius:15px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # ========================= FEATURE CONTRIBUTION BAR (SMALL + GRADIENT) =========================
    st.markdown("---")
    st.subheader("🔬 Feature Contribution to Risk Score")
    feature_values = [age, bmi, blood_pressure, cholesterol, glucose, heart_rate]

    # Normalize for gradient
    max_val = max(feature_values)
    min_val = min(feature_values)
    norm_values = [(v - min_val)/(max_val - min_val + 1e-6) for v in feature_values]
    cmap = cm.get_cmap('RdYlGn_r')
    colors_list = [cmap(v) for v in norm_values]
    top_idx = np.argmax(feature_values)
    colors_list[top_idx] = '#FF9800'

    fig, ax = plt.subplots(figsize=(5,2.5))  # smaller chart
    ax.barh(feature_names, feature_values, color=colors_list)
    ax.set_xlabel("Value")
    ax.set_title("Feature Contributions to Risk")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # ========================= PERSONALIZED RECOMMENDATION =========================
    top_feature = feature_names[top_idx]
    top_value = feature_values[top_idx]
    recommendations = {
        "Age": "Ensure regular health checkups and maintain an active lifestyle.",
        "BMI": "Maintain a balanced diet and exercise regularly to control BMI.",
        "BloodPressure": "Monitor blood pressure frequently and reduce salt intake.",
        "Cholesterol": "Limit saturated fats and get regular lipid profile tests.",
        "Glucose": "Maintain a healthy diet, monitor blood sugar levels, and exercise.",
        "HeartRate": "Engage in cardiovascular exercise and monitor heart health."
    }
    rec_text = recommendations.get(top_feature, "Maintain healthy lifestyle habits.")
    st.markdown("---")
    st.subheader("💡 Personalized Recommendation")
    st.info(f"⚡ Based on your **{top_feature} ({top_value})**, recommendation: {rec_text}")

    # ========================= PDF REPORT =========================
    st.markdown("---")
    st.subheader("📑 Download Medical Report")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Risk image for PDF
    response = requests.get(risk_img_url)
    img_bytes = BytesIO(response.content)
    elements.append(RLImage(img_bytes, width=200, height=120))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("MediTrack Health Report", styles['Title']))
    elements.append(Spacer(1,0.2*inch))

    # Patient table
    table_data = [["Feature","Value"]] + [[k,v] for k,v in patient_summary.items()]
    table = Table(table_data, colWidths=[3*inch,3*inch])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightblue),
                               ('GRID',(0,0),(-1,-1),1,colors.grey),
                               ('ALIGN',(0,0),(-1,-1),'LEFT'),
                               ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                               ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke)]))
    elements.append(table)
    elements.append(Spacer(1,0.2*inch))

    # Feature contribution chart in PDF
    fig_pdf, ax_pdf = plt.subplots(figsize=(5,2.5))
    ax_pdf.barh(feature_names, feature_values, color=colors_list)
    ax_pdf.set_xlabel("Value")
    ax_pdf.set_title("Feature Contributions to Risk")
    plt.tight_layout()
    img_buffer = BytesIO()
    fig_pdf.savefig(img_buffer, format='PNG')
    plt.close(fig_pdf)
    img_buffer.seek(0)
    pdf_img = RLImage(img_buffer, width=350, height=180)
    elements.append(Paragraph("Feature Contribution to Risk", styles['Heading2']))
    elements.append(Spacer(1,0.1*inch))
    elements.append(pdf_img)
    elements.append(Spacer(1,0.2*inch))

    # Prediction, diagnosis & recommendation
    elements.append(Paragraph("Prediction Results", styles['Heading2']))
    elements.append(Paragraph(f"Final Risk Score: {avg_risk:.2f}", styles['Normal']))
    elements.append(Paragraph(f"Risk Level: {risk_level}", styles['Normal']))
    elements.append(Paragraph(f"Diagnosis: {diagnosis}", styles['Normal']))
    elements.append(Paragraph(f"Recommendation: {rec_text}", styles['Normal']))

    if doctor_notes.strip()!="":
        elements.append(Spacer(1,0.1*inch))
        elements.append(Paragraph("Doctor Notes", styles['Heading2']))
        elements.append(Paragraph(doctor_notes, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)

    st.download_button("⬇️ Download PDF Report", buffer, file_name="MediTrack_Report.pdf", mime="application/pdf")
