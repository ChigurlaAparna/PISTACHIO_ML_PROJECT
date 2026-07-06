import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import time

st.set_page_config(page_title="Pistachio Classification System", page_icon="🥜", layout="wide", initial_sidebar_state="collapsed")

PISTACHIO_SVG = """
<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="shellGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#8BC34A"/><stop offset="50%" style="stop-color:#689F38"/><stop offset="100%" style="stop-color:#558B2F"/>
</linearGradient>
<linearGradient id="nutGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#FFEB3B"/><stop offset="50%" style="stop-color:#FFC107"/><stop offset="100%" style="stop-color:#FF9800"/>
</linearGradient>
</defs>
<ellipse cx="50" cy="55" rx="32" ry="38" fill="url(#shellGrad)"/>
<path d="M 50 20 Q 45 35, 50 55 Q 55 35, 50 20" fill="#558B2F" opacity="0.6"/>
<ellipse cx="50" cy="52" rx="18" ry="22" fill="url(#nutGrad)"/>
<ellipse cx="44" cy="45" rx="6" ry="8" fill="#FFEB3B" opacity="0.5"/>
</svg>
"""

FEATURES = ["AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS", "ECCENTRICITY", "EQDIASQ", "SOLIDITY", "CONVEX_AREA", "EXTENT", "ASPECT_RATIO", "ROUNDNESS", "COMPACTNESS", "SHAPEFACTOR_1", "SHAPEFACTOR_2", "SHAPEFACTOR_3", "SHAPEFACTOR_4"]

FEATURE_EXPLANATION = {
    "AREA": "Total size in pixels", "PERIMETER": "Outer boundary length", "MAJOR_AXIS": "Longest diameter", "MINOR_AXIS": "Shortest diameter",
    "ECCENTRICITY": "How elongated (1=line, 0=circle)", "EQDIASQ": "Circle with same area", "SOLIDITY": "Area/convex area ratio", "CONVEX_AREA": "Convex hull area",
    "EXTENT": "Area/bounding box ratio", "ASPECT_RATIO": "Major/Minor axis", "ROUNDNESS": "Circularity (1=circle)", "COMPACTNESS": "Compactness measure",
    "SHAPEFACTOR_1": "Normalized area ratio", "SHAPEFACTOR_2": "Normalized area ratio", "SHAPEFACTOR_3": "Normalized ratio", "SHAPEFACTOR_4": "Normalized ratio"
}

FEATURE_RANGES = {
    "AREA": (20000, 150000), "PERIMETER": (500, 2500), "MAJOR_AXIS": (100, 800), "MINOR_AXIS": (50, 500),
    "ECCENTRICITY": (0.1, 0.99), "EQDIASQ": (100, 600), "SOLIDITY": (0.7, 1.0), "CONVEX_AREA": (20000, 160000),
    "EXTENT": (0.3, 0.95), "ASPECT_RATIO": (0.5, 4.0), "ROUNDNESS": (0.3, 1.0), "COMPACTNESS": (0.3, 1.0),
    "SHAPEFACTOR_1": (0.001, 0.02), "SHAPEFACTOR_2": (0.001, 0.02), "SHAPEFACTOR_3": (0.1, 0.99), "SHAPEFACTOR_4": (0.1, 0.99)
}

EXAMPLE_VALUES = {"AREA": 85831, "PERIMETER": 1185.4930, "MAJOR_AXIS": 448.8785, "MINOR_AXIS": 244.3473, "ECCENTRICITY": 0.8389, "EQDIASQ": 330.5804, "SOLIDITY": 0.9823, "CONVEX_AREA": 87377, "EXTENT": 0.7485, "ASPECT_RATIO": 1.8371, "ROUNDNESS": 0.7675, "COMPACTNESS": 0.7365, "SHAPEFACTOR_1": 0.0052, "SHAPEFACTOR_2": 0.0028, "SHAPEFACTOR_3": 0.5424, "SHAPEFACTOR_4": 0.9964}

for f in FEATURES:
    if f not in st.session_state: st.session_state[f] = 0.0
if "result" not in st.session_state: st.session_state.result = None
if "confidence" not in st.session_state: st.session_state.confidence = None
if "probabilities" not in st.session_state: st.session_state.probabilities = None
if "history" not in st.session_state: st.session_state.history = []
if "page" not in st.session_state: st.session_state.page = "Home"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
#MainMenu, footer, header {visibility:hidden;}
* {font-family: 'Poppins', sans-serif;}
.stApp {background: radial-gradient(circle at 15% 10%, rgba(191,231,116,0.65), transparent 28%), radial-gradient(circle at 90% 18%, rgba(250,220,106,0.58), transparent 24%), linear-gradient(135deg,#edf7e8 0%,#dcefd5 50%,#eaf6d4 100%); min-height: 100vh;}
.particles {position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; overflow: hidden; z-index: 0;}
.particle {position: absolute; width: 8px; height: 8px; background: linear-gradient(135deg, #7cb342, #aee02f); border-radius: 50%; animation: float 25s infinite; box-shadow: 0 0 15px rgba(174, 224, 47, 0.6);}
.particle:nth-child(1){left:10%;animation-delay:0s;animation-duration:25s;}.particle:nth-child(2){left:20%;animation-delay:2s;animation-duration:20s;}.particle:nth-child(3){left:30%;animation-delay:4s;animation-duration:28s;}.particle:nth-child(4){left:40%;animation-delay:1s;animation-duration:22s;}.particle:nth-child(5){left:50%;animation-delay:3s;animation-duration:26s;}.particle:nth-child(6){left:60%;animation-delay:5s;animation-duration:24s;}.particle:nth-child(7){left:70%;animation-delay:0.5s;animation-duration:21s;}.particle:nth-child(8){left:80%;animation-delay:2.5s;animation-duration:27s;}.particle:nth-child(9){left:90%;animation-delay:4.5s;animation-duration:23s;}.particle:nth-child(10){left:15%;animation-delay:1.5s;animation-duration:29s;}
@keyframes float {0%,100%{transform:translateY(100vh) rotate(0deg);opacity:0;}10%{opacity:1;}90%{opacity:1;}100%{transform:translateY(-100vh) rotate(720deg);opacity:0;}}
.block-container {max-width: 1200px; padding-top: 20px; padding-bottom: 50px; position: relative; z-index: 1;}
.glass-card {background: rgba(255, 255, 255, 0.87); backdrop-filter: blur(20px); border: 1px solid rgba(124, 179, 66, 0.3); border-radius: 28px; padding: 26px; box-shadow: 0 14px 35px rgba(37, 82, 42, 0.14); transition: all 0.4s ease;}
.glass-card:hover {transform: translateY(-5px); box-shadow: 0 20px 40px rgba(37, 82, 42, 0.2);}
.hero {text-align: center; padding: 10px 10px 35px 10px;}
.logo-wrap {width: 130px; height: 130px; margin: 0 auto 25px auto; border-radius: 50%; background: linear-gradient(145deg, #efb65e, #d98b35); display: flex; align-items: center; justify-content: center; box-shadow: 0 15px 35px rgba(109, 77, 20, 0.35), 0 0 60px rgba(239, 182, 94, 0.4); animation: floatLogo 3s ease-in-out infinite, pulse-glow 2s ease-in-out infinite; position: relative;}
.logo-wrap::before {content: ''; position: absolute; width: 150px; height: 150px; border-radius: 50%; border: 2px solid rgba(174, 224, 47, 0.5); animation: ripple 2.5s ease-out infinite;}
.logo-wrap::after {content: ''; position: absolute; width: 170px; height: 170px; border-radius: 50%; border: 1px solid rgba(174, 224, 47, 0.3); animation: ripple 2.5s ease-out infinite 0.6s;}
@keyframes ripple {0%{transform:scale(1);opacity:0.8;}100%{transform:scale(1.25);opacity:0;}}
@keyframes floatLogo {0%,100%{transform:translateY(0);}50%{transform:translateY(-12px);}}
@keyframes pulse-glow {0%,100%{box-shadow:0 15px 35px rgba(109,77,20,0.35),0 0 60px rgba(239,182,94,0.4);}50%{box-shadow:0 15px 35px rgba(109,77,20,0.5),0 0 80px rgba(239,182,94,0.6);}}
.pistachio-svg {width: 90px; height: 90px; animation: wobble 2.5s ease-in-out infinite;}
@keyframes wobble {0%,100%{transform:rotate(-5deg) scale(1);}25%{transform:rotate(-10deg) scale(1.05);}75%{transform:rotate(5deg) scale(1.05);}}
.badge {display: inline-block; padding: 10px 24px; border-radius: 50px; background: rgba(228, 246, 198, 0.9); border: 1px solid #c4e777; color: #315f36; font-size: 13px; font-weight: 800; margin-bottom: 20px; animation: badge-pulse 2.5s ease-in-out infinite;}
@keyframes badge-pulse {0%,100%{transform:scale(1);}50%{transform:scale(1.03);}}
.hero-title {font-size: 68px; line-height: 1.05; font-weight: 900; color: #1e5a2c; animation: title-reveal 1s ease-out;}
.hero-title span {color: #aee02f; display: block;}
@keyframes title-reveal {0%{opacity:0;transform:translateY(30px);}100%{opacity:1;transform:translateY(0);}}
.hero-description {max-width: 700px; margin: 20px auto 0 auto; font-size: 17px; line-height: 1.7; color: #3d6045; animation: fade-up 1s ease-out 0.3s both;}
@keyframes fade-up {0%{opacity:0;transform:translateY(20px);}100%{opacity:1;transform:translateY(0);}}
.stats-row {display: flex; justify-content: center; gap: 22px; flex-wrap: wrap; margin-top: 40px;}
.stat-box {width: 150px; padding: 22px 12px; border-radius: 22px; background: rgba(255, 255, 255, 0.82); box-shadow: 0 12px 25px rgba(45, 93, 49, 0.12); animation: stat-enter 0.6s ease-out both;}
.stat-box:nth-child(1){animation-delay:0.1s;}.stat-box:nth-child(2){animation-delay:0.2s;}.stat-box:nth-child(3){animation-delay:0.3s;}.stat-box:nth-child(4){animation-delay:0.4s;}
@keyframes stat-enter {0%{opacity:0;transform:translateY(30px) scale(0.8);}100%{opacity:1;transform:translateY(0) scale(1);}}
.stat-box:hover {transform:translateY(-8px) scale(1.05); box-shadow: 0 18px 35px rgba(45, 93, 49, 0.18);}
.stat-value {font-size: 30px; font-weight: 900; color: #173d25;}
.stat-label {margin-top: 6px; font-size: 11px; font-weight: 800; letter-spacing: 1.5px; color: #6b8d6f;}
.section-heading {text-align: center; font-size: 30px; font-weight: 800; color: #1d542b; margin: 10px 0 6px 0;}
.section-subheading {text-align: center; color: #607763; margin-bottom: 20px; font-size: 14px;}
.stButton > button {width: 100%; border: none; border-radius: 14px; padding: 14px 20px; font-weight: 700; font-size: 14px; background: linear-gradient(135deg, #123d24, #235e35); color: white; position: relative; overflow: hidden; transition: all 0.4s ease; box-shadow: 0 6px 20px rgba(18, 61, 36, 0.3);}
.stButton > button::before {content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); transition: left 0.6s ease;}
.stButton > button:hover::before {left: 100%;}
.stButton > button:hover {transform: translateY(-3px); box-shadow: 0 12px 30px rgba(18, 61, 36, 0.4);}
.feature-label {display: flex; align-items: center; gap: 10px; margin-top: 12px; margin-bottom: 6px;}
.feature-number {width: 28px; height: 28px; border-radius: 8px; background: #17472a; color: white; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 900; box-shadow: 0 4px 12px rgba(23, 71, 42, 0.3);}
.feature-number:hover {transform: scale(1.1) rotate(5deg); background: #2d7a46;}
.feature-name {color: #173d25; font-size: 13px; font-weight: 700;}
.feature-help {color: #718175; font-size: 11px;}
.stNumberInput > div > div > input {background: rgba(255, 255, 255, 0.9) !important; border: 1px solid rgba(45, 93, 49, 0.2) !important; border-radius: 12px !important; color: #173d25 !important; padding: 12px 16px !important; font-size: 14px !important;}
.stNumberInput > label {display: none;}
.result-card {margin-top: 25px; padding: 30px; text-align: center; background: linear-gradient(135deg, #e8f7d4, #ffffff); border: 2px solid #b9df58; border-radius: 24px; animation: result-reveal 0.8s ease-out;}
@keyframes result-reveal {0%{opacity:0;transform:scale(0.8) translateY(20px);}100%{opacity:1;transform:scale(1) translateY(0);}}
.result-card .result-text {font-size: 44px; font-weight: 900; color: #1c602d; margin: 12px 0;}
.result-card .confidence-text {font-size: 18px; font-weight: 700; color: #315f36;}
.confidence-bar {width: 100%; height: 10px; background: rgba(45, 93, 49, 0.1); border-radius: 10px; margin-top: 15px; overflow: hidden;}
.confidence-fill {height: 100%; background: linear-gradient(90deg, #7cb342, #aee02f); border-radius: 10px; animation: confidence-grow 1.5s ease-out;}
@keyframes confidence-grow {0%{width:0%;}}
.info-card {background: rgba(255, 255, 255, 0.87); border: 1px solid rgba(124, 179, 66, 0.3); border-radius: 28px; padding: 26px;}
.model-item {padding: 16px; background: #f2f8eb; border-radius: 16px; margin-top: 12px; border: 1px solid rgba(45, 93, 49, 0.1);}
.model-item:hover {transform: translateX(5px); background: #e8f5df;}
.model-label {font-size: 10px; font-weight: 900; color: #728074; letter-spacing: 1.5px;}
.model-value {font-size: 15px; font-weight: 800; color: #173d25; margin-top: 4px;}
.tip-box {margin-top: 20px; padding: 16px; background: #f3fae6; border-left: 4px solid #9fd433; border-radius: 10px; color: #3e5d43;}
.card-title {font-size: 22px; font-weight: 800; color: #173d25;}
.card-subtitle {font-size: 13px; color: #718175; margin-top: 6px;}
.form-card {background: rgba(255, 255, 255, 0.87); border-radius: 28px; padding: 24px; box-shadow: 0 14px 35px rgba(37, 82, 42, 0.12); margin-bottom: 12px;}
.metric-card {background: rgba(255, 255, 255, 0.87); border: 1px solid rgba(124, 179, 66, 0.3); border-radius: 20px; padding: 24px; text-align: center; transition: all 0.3s ease;}
.metric-card:hover {transform: translateY(-5px);}
.metric-value {font-size: 36px; font-weight: 900; color: #1c602d;}
.metric-label {font-size: 12px; color: #607763; margin-top: 8px; text-transform: uppercase; letter-spacing: 1px;}
.footer-text {text-align: center; margin-top: 45px; padding: 20px; color: #55745a; font-size: 13px;}
.footer-text b {color: #315f36;}
.nav-btn {background: rgba(255,255,255,0.8) !important; border: 1px solid rgba(124,179,66,0.3) !important; color: #315f36 !important; padding: 12px 20px !important; border-radius: 12px !important; transition: all 0.3s ease !important;}
.nav-btn:hover {background: rgba(174,224,47,0.3) !important; border-color: #aee02f !important;}
.nav-btn.active {background: #aee02f !important; color: #1c602d !important;}
::-webkit-scrollbar {width: 10px;}::-webkit-scrollbar-track {background: rgba(255,255,255,0.3);}::-webkit-scrollbar-thumb {background: rgba(174,224,47,0.5); border-radius: 5px;}
@media (max-width: 768px) {.hero-title{font-size:42px;}.stats-row{gap:15px;}.stat-box{width:130px;}}
</style>
"""

@st.cache_resource
def load_model():
    base = Path(__file__).parent
    return joblib.load(base/"pistachio_gradient_boosting_model.pkl"), joblib.load(base/"pistachio_scaler.pkl"), joblib.load(base/"pistachio_label_encoder.pkl")

def predict():
    model, scaler, encoder = load_model()
    vals = [st.session_state[f] for f in FEATURES]
    df = pd.DataFrame([vals], columns=FEATURES)
    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    probs = model.predict_proba(scaled)[0]
    label = encoder.inverse_transform(pred)[0]
    return label, float(max(probs)*100), float(probs[0]), float(probs[1])

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="particles"><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
if c1.button("🏠 Home", key="nav_home", use_container_width=True): st.session_state.page = "Home"
if c2.button("🔮 Predict", key="nav_predict", use_container_width=True): st.session_state.page = "Predict"
if c3.button("📊 Model", key="nav_model", use_container_width=True): st.session_state.page = "Model"
if c4.button("📋 Dataset", key="nav_dataset", use_container_width=True): st.session_state.page = "Dataset"
st.write("")

if st.session_state.page == "Home":
    st.markdown(f'<div class="hero"><div class="logo-wrap"><div class="pistachio-svg">{PISTACHIO_SVG}</div></div><div class="badge">⭐ AI-Powered Classification</div><div class="hero-title">Pistachio<span>Classification System</span></div><div class="hero-description">Advanced machine learning classification of Kirmizi and Siirt pistachio varieties using 16 morphological measurements.</div><div class="stats-row"><div class="stat-box"><div class="stat-value">GB</div><div class="stat-label">Algorithm</div></div><div class="stat-box"><div class="stat-value">16</div><div class="stat-label">Features</div></div><div class="stat-box"><div class="stat-value">2</div><div class="stat-label">Varieties</div></div><div class="stat-box"><div class="stat-value">{datetime.now().strftime("%H:%M")}</div><div class="stat-label">Session</div></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">How It Works</div><div class="section-subheading">4 simple steps</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (icon, step, title, desc) in enumerate([("🥜","Step 1","Pistachio Image","Capture"),("📐","Step 2","Feature Extraction","16 measurements"),("⚙️","Step 3","StandardScaler","Normalization"),("🎯","Step 4","GB Classifier","Kirmizi / Siirt")]):
        with cols[i]:
            st.markdown(f'<div class="glass-card" style="text-align:center;"><div style="font-size:48px;margin-bottom:15px;">{icon}</div><div style="font-size:18px;font-weight:700;color:#1e5a2c;">{step}</div><div style="font-size:14px;color:#3d6045;margin-top:10px;">{title}</div><div style="font-size:12px;color:#607763;margin-top:5px;">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer-text">🥜 <b>Pistachio Classification System</b><br>Built with Python, Streamlit and Machine Learning</div>', unsafe_allow_html=True)

elif st.session_state.page == "Predict":
    st.markdown('<div class="section-heading">Enter Pistachio Measurements</div><div class="section-subheading">Fill in 16 morphological features</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if b1.button("✨ Load Example Values", use_container_width=True):
        for f in FEATURES: st.session_state[f] = EXAMPLE_VALUES[f]
    if b2.button("↻ Reset Values", use_container_width=True):
        for f in FEATURES: st.session_state[f] = 0.0
    st.write("")
    m1, m2 = st.columns([1.2, 0.8], gap="large")
    with m1:
        st.markdown('<div class="form-card"><div class="card-title">📊 Enter Measurements</div></div>', unsafe_allow_html=True)
        lc, rc = st.columns(2, gap="medium")
        for i, f in enumerate(FEATURES):
            with (lc if i % 2 == 0 else rc):
                st.markdown(f'<div class="feature-label"><span class="feature-number">{i+1}</span><div><div class="feature-name">{f}</div><div class="feature-help">{FEATURE_EXPLANATION[f]}</div></div></div>', unsafe_allow_html=True)
                st.number_input(f, min_value=0.0, step=0.0001, format="%.4f", key=f, label_visibility="collapsed")
        with st.expander("📖 Feature Explanations"):
            for f, e in FEATURE_EXPLANATION.items():
                st.markdown(f"**{f}**: {e}")
        st.write("")
        if st.button("🔮 Classify Pistachio", use_container_width=True):
            empty = [f for f in FEATURES if st.session_state[f] == 0]
            if empty:
                st.error(f"⚠️ Enter values for: {', '.join(empty)}")
            else:
                bar = st.progress(0)
                for i in range(50):
                    time.sleep(0.01)
                    bar.progress(i+1)
                try:
                    label, conf, k, s = predict()
                    st.session_state.result = label
                    st.session_state.confidence = conf
                    st.session_state.probabilities = [k, s]
                    st.session_state.history.insert(0, {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M"), "result": label, "confidence": f"{conf:.1f}%", "kirmizi": f"{k*100:.1f}%", "siirt": f"{s*100:.1f}%"})
                    if len(st.session_state.history) > 10: st.session_state.history = st.session_state.history[:10]
                except Exception as e:
                    st.error(f"Error: {e}")
                bar.empty()
    with m2:
        st.markdown('<div class="info-card"><div class="card-title">ℹ️ Model Information</div><div class="card-subtitle">Classification details</div><div class="model-item"><div class="model-label">🧠 ALGORITHM</div><div class="model-value">Gradient Boosting</div></div><div class="model-item"><div class="model-label">⚙️ PREPROCESSING</div><div class="model-value">StandardScaler</div></div><div class="model-item"><div class="model-label">🎯 OUTPUT</div><div class="model-value">Kirmizi / Siirt</div></div><div class="tip-box">💡 Use image analysis to extract measurements.</div></div>', unsafe_allow_html=True)
        if st.session_state.result:
            st.markdown(f'<div class="result-card"><div style="font-size:11px;font-weight:700;color:#718175;letter-spacing:2px;">PREDICTED VARIETY</div><div class="result-text">{st.session_state.result}</div><div class="confidence-text">Confidence: {st.session_state.confidence:.1f}%</div><div class="confidence-bar"><div class="confidence-fill" style="width:{st.session_state.confidence}%;"></div></div></div>', unsafe_allow_html=True)
            st.markdown("### 📈 Probability Distribution")
            st.bar_chart({"Kirmizi": st.session_state.probabilities[0]*100, "Siirt": st.session_state.probabilities[1]*100})
            data = {"Feature": FEATURES, "Value": [st.session_state[f] for f in FEATURES]}
            csv = pd.DataFrame(data).to_csv(index=False)
            st.download_button("📥 Download Report", csv, "prediction.csv", "text/csv", use_container_width=True)
    if st.session_state.history:
        with st.expander("📜 Prediction History"):
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
            if st.button("🗑️ Clear History"): st.session_state.history = []

elif st.session_state.page == "Model":
    st.markdown('<div class="section-heading">Model Information</div><div class="section-subheading">Technical details</div>', unsafe_allow_html=True)
    c = st.columns(4)
    for i, (v, l) in enumerate([("GB","Algorithm"),("StandardScaler","Preprocessing"),("Binary","Model Type"),("16","Features")]):
        with c[i]: st.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="glass-card"><div class="card-title">🌳 Gradient Boosting</div><p style="margin-top:15px;color:#3d6045;">Ensemble learning that builds models sequentially. Each new model corrects errors from previous ones.</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="glass-card"><div class="card-title">📏 StandardScaler</div><p style="margin-top:15px;color:#3d6045;">Scales features to unit variance. Ensures all 16 features contribute equally.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Model Performance</div><div class="section-subheading">Validation metrics</div>', unsafe_allow_html=True)
    c = st.columns(5)
    for i, (v, l) in enumerate([("87.9%","Accuracy"),("83.52%","Precision"),("88.52%","Recall"),("85.95%","F1 Score"),("1000","Val. Set")]):
        with c[i]: st.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

elif st.session_state.page == "Dataset":
    st.markdown('<div class="section-heading">About Dataset</div><div class="section-subheading">Understanding the system</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><div class="card-title">🎯 What is this?</div><p style="margin-top:15px;color:#3d6045;font-size:16px;line-height:1.8;">Classifies pistachios into <b>Kirmizi</b> and <b>Siirt</b> varieties.</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="glass-card"><div class="card-title">🥜 Kirmizi</div><p style="margin-top:15px;color:#3d6045;">Known for bright green color and rich flavor.</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="glass-card"><div class="card-title">🥜 Siirt</div><p style="margin-top:15px;color:#3d6045;">Traditional Turkish variety with unique taste.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="glass-card"><div class="card-title">📋 Workflow</div><p style="margin-top:15px;color:#3d6045;"><b>Note:</b> Uses numerical measurements, not raw images.</p><ol style="margin-top:20px;color:#3d6045;line-height:2;"><li><b>Image Capture</b></li><li><b>Feature Extraction</b> - 16 measurements</li><li><b>Preprocessing</b> - StandardScaler</li><li><b>Classification</b> - GB predicts variety</li><li><b>Result</b> - Display with confidence</li></ol></div>', unsafe_allow_html=True)