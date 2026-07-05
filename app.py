import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import time


st.set_page_config(
    page_title="Pistachio Classification System",
    page_icon="🥜",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Custom SVG Green Pistachio Logo
PISTACHIO_LOGO_SVG = """
<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="shellGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#8BC34A;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#689F38;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#558B2F;stop-opacity:1" />
        </linearGradient>
        <linearGradient id="nutGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#FFEB3B;stop-opacity:1" />
            <stop offset="50%" style="stop-color:#FFC107;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#FF9800;stop-opacity:1" />
        </linearGradient>
        <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        </filter>
    </defs>
    <!-- Outer shell -->
    <ellipse cx="50" cy="55" rx="32" ry="38" fill="url(#shellGrad)" filter="url(#glow)"/>
    <!-- Shell opening crack -->
    <path d="M 50 20 Q 45 35, 50 55 Q 55 35, 50 20" fill="#558B2F" opacity="0.6"/>
    <!-- Inner nut -->
    <ellipse cx="50" cy="52" rx="18" ry="22" fill="url(#nutGrad)"/>
    <!-- Nut highlight -->
    <ellipse cx="44" cy="45" rx="6" ry="8" fill="#FFEB3B" opacity="0.5"/>
    <!-- Shell edge highlight -->
    <path d="M 25 45 Q 20 55, 30 70" stroke="#AED581" stroke-width="2" fill="none" opacity="0.6"/>
</svg>
"""


FEATURES = [
    "AREA", "PERIMETER", "MAJOR_AXIS", "MINOR_AXIS",
    "ECCENTRICITY", "EQDIASQ", "SOLIDITY", "CONVEX_AREA",
    "EXTENT", "ASPECT_RATIO", "ROUNDNESS", "COMPACTNESS",
    "SHAPEFACTOR_1", "SHAPEFACTOR_2", "SHAPEFACTOR_3", "SHAPEFACTOR_4"
]


FEATURE_HELP = {
    "AREA": "Pixel count",
    "PERIMETER": "Edge length",
    "MAJOR_AXIS": "Long diameter",
    "MINOR_AXIS": "Short diameter",
    "ECCENTRICITY": "Shape deviation",
    "EQDIASQ": "Equivalent diameter",
    "SOLIDITY": "Solidity ratio",
    "CONVEX_AREA": "Convex hull area",
    "EXTENT": "Extent ratio",
    "ASPECT_RATIO": "Length ratio",
    "ROUNDNESS": "Circularity",
    "COMPACTNESS": "Compactness",
    "SHAPEFACTOR_1": "SF #1",
    "SHAPEFACTOR_2": "SF #2",
    "SHAPEFACTOR_3": "SF #3",
    "SHAPEFACTOR_4": "SF #4"
}


EXAMPLE_VALUES = {
    "AREA": 85831.0000,
    "PERIMETER": 1185.4930,
    "MAJOR_AXIS": 448.8785,
    "MINOR_AXIS": 244.3473,
    "ECCENTRICITY": 0.8389,
    "EQDIASQ": 330.5804,
    "SOLIDITY": 0.9823,
    "CONVEX_AREA": 87377.0000,
    "EXTENT": 0.7485,
    "ASPECT_RATIO": 1.8371,
    "ROUNDNESS": 0.7675,
    "COMPACTNESS": 0.7365,
    "SHAPEFACTOR_1": 0.0052,
    "SHAPEFACTOR_2": 0.0028,
    "SHAPEFACTOR_3": 0.5424,
    "SHAPEFACTOR_4": 0.9964
}


# ---------- SESSION STATE ----------
for feature in FEATURES:
    if feature not in st.session_state:
        st.session_state[feature] = 0.0


if "result" not in st.session_state:
    st.session_state.result = None


if "confidence" not in st.session_state:
    st.session_state.confidence = None


if "is_loading" not in st.session_state:
    st.session_state.is_loading = False


if "animation_trigger" not in st.session_state:
    st.session_state.animation_trigger = 0




def load_example_values():
    for feature in FEATURES:
        st.session_state[feature] = EXAMPLE_VALUES[feature]
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.animation_trigger += 1




def reset_all_values():
    for feature in FEATURES:
        st.session_state[feature] = 0.0
    st.session_state.result = None
    st.session_state.confidence = None
    st.session_state.animation_trigger += 1




@st.cache_resource
def load_model_files():
    base_path = Path(__file__).parent
    model = joblib.load(base_path / "pistachio_gradient_boosting_model.pkl")
    scaler = joblib.load(base_path / "pistachio_scaler.pkl")
    encoder = joblib.load(base_path / "pistachio_label_encoder.pkl")
    return model, scaler, encoder




# ---------- ENHANCED CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(191,231,116,0.65), transparent 28%),
        radial-gradient(circle at 90% 18%, rgba(250,220,106,0.58), transparent 24%),
        linear-gradient(135deg,#edf7e8 0%,#dcefd5 50%,#eaf6d4 100%);
    min-height: 100vh;
}

/* Animated particles background */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    overflow: hidden;
    z-index: 0;
}

.particle {
    position: absolute;
    width: 8px;
    height: 8px;
    background: linear-gradient(135deg, #7cb342, #aee02f);
    border-radius: 50%;
    animation: float 25s infinite;
    box-shadow: 0 0 15px rgba(174, 224, 47, 0.6);
}

.particle:nth-child(1) { left: 10%; animation-delay: 0s; animation-duration: 25s; }
.particle:nth-child(2) { left: 20%; animation-delay: 2s; animation-duration: 20s; }
.particle:nth-child(3) { left: 30%; animation-delay: 4s; animation-duration: 28s; }
.particle:nth-child(4) { left: 40%; animation-delay: 1s; animation-duration: 22s; }
.particle:nth-child(5) { left: 50%; animation-delay: 3s; animation-duration: 26s; }
.particle:nth-child(6) { left: 60%; animation-delay: 5s; animation-duration: 24s; }
.particle:nth-child(7) { left: 70%; animation-delay: 0.5s; animation-duration: 21s; }
.particle:nth-child(8) { left: 80%; animation-delay: 2.5s; animation-duration: 27s; }
.particle:nth-child(9) { left: 90%; animation-delay: 4.5s; animation-duration: 23s; }
.particle:nth-child(10) { left: 15%; animation-delay: 1.5s; animation-duration: 29s; }

@keyframes float {
    0%, 100% {
        transform: translateY(100vh) rotate(0deg);
        opacity: 0;
    }
    10% {
        opacity: 1;
    }
    90% {
        opacity: 1;
    }
    100% {
        transform: translateY(-100vh) rotate(720deg);
        opacity: 0;
    }
}

.block-container {
    max-width: 1300px;
    padding-top: 35px;
    padding-bottom: 50px;
    position: relative;
    z-index: 1;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.87);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(124, 179, 66, 0.3);
    border-radius: 28px;
    padding: 26px;
    box-shadow: 
        0 14px 35px rgba(37, 82, 42, 0.14),
        inset 0 1px 0 rgba(255, 255, 255, 0.8);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 20px 40px rgba(37, 82, 42, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.9);
    border-color: rgba(174, 224, 47, 0.5);
}

/* Hero section */
.hero {
    text-align: center;
    padding: 10px 10px 35px 10px;
}

.logo-wrap {
    width: 130px;
    height: 130px;
    margin: 0 auto 25px auto;
    border-radius: 50%;
    background: linear-gradient(145deg, #efb65e, #d98b35);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 
        0 15px 35px rgba(109, 77, 20, 0.35),
        0 0 60px rgba(239, 182, 94, 0.4);
    animation: floatLogo 3s ease-in-out infinite, pulse-glow 2s ease-in-out infinite;
    position: relative;
}

.logo-wrap::before {
    content: '';
    position: absolute;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    border: 2px solid rgba(174, 224, 47, 0.5);
    animation: ripple 2.5s ease-out infinite;
}

.logo-wrap::after {
    content: '';
    position: absolute;
    width: 170px;
    height: 170px;
    border-radius: 50%;
    border: 1px solid rgba(174, 224, 47, 0.3);
    animation: ripple 2.5s ease-out infinite 0.6s;
}

@keyframes ripple {
    0% {
        transform: scale(1);
        opacity: 0.8;
    }
    100% {
        transform: scale(1.25);
        opacity: 0;
    }
}

@keyframes floatLogo {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 15px 35px rgba(109, 77, 20, 0.35), 0 0 60px rgba(239, 182, 94, 0.4); }
    50% { box-shadow: 0 15px 35px rgba(109, 77, 20, 0.5), 0 0 80px rgba(239, 182, 94, 0.6); }
}

.pistachio-svg {
    width: 90px;
    height: 90px;
    animation: wobble 2.5s ease-in-out infinite;
}

@keyframes wobble {
    0%, 100% { transform: rotate(-5deg) scale(1); }
    25% { transform: rotate(-10deg) scale(1.05); }
    75% { transform: rotate(5deg) scale(1.05); }
}

.badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 50px;
    background: rgba(228, 246, 198, 0.9);
    border: 1px solid #c4e777;
    color: #315f36;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 20px;
    animation: badge-pulse 2.5s ease-in-out infinite;
    letter-spacing: 0.5px;
}

@keyframes badge-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.03); }
}

.hero-title {
    font-size: 68px;
    line-height: 1.05;
    font-weight: 900;
    color: #1e5a2c;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    animation: title-reveal 1s ease-out;
}

@keyframes title-reveal {
    0% {
        opacity: 0;
        transform: translateY(30px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-title span {
    color: #aee02f;
    display: block;
}

.hero-description {
    max-width: 700px;
    margin: 20px auto 0 auto;
    font-size: 17px;
    line-height: 1.7;
    color: #3d6045;
    animation: fade-up 1s ease-out 0.3s both;
}

@keyframes fade-up {
    0% {
        opacity: 0;
        transform: translateY(20px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 22px;
    flex-wrap: wrap;
    margin-top: 40px;
}

.stat-box {
    width: 150px;
    padding: 22px 12px;
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.82);
    box-shadow: 0 12px 25px rgba(45, 93, 49, 0.12);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: stat-enter 0.6s ease-out both;
}

.stat-box:nth-child(1) { animation-delay: 0.1s; }
.stat-box:nth-child(2) { animation-delay: 0.2s; }
.stat-box:nth-child(3) { animation-delay: 0.3s; }
.stat-box:nth-child(4) { animation-delay: 0.4s; }

@keyframes stat-enter {
    0% {
        opacity: 0;
        transform: translateY(30px) scale(0.8);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.stat-box:hover {
    transform: translateY(-8px) scale(1.05);
    box-shadow: 0 18px 35px rgba(45, 93, 49, 0.18);
}

.stat-value {
    font-size: 30px;
    font-weight: 900;
    color: #173d25;
}

.stat-label {
    margin-top: 6px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    color: #6b8d6f;
}

/* Section headings */
.section-heading {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: #1d542b;
    margin: 10px 0 6px 0;
    animation: fade-up 0.8s ease-out;
}

.section-subheading {
    text-align: center;
    color: #607763;
    margin-bottom: 20px;
    font-size: 14px;
    animation: fade-up 0.8s ease-out 0.1s both;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 14px 20px;
    font-weight: 800;
    font-size: 14px;
    background: linear-gradient(135deg, #123d24, #235e35);
    color: white;
    position: relative;
    overflow: hidden;
    transition: all 0.4s ease;
    box-shadow: 0 6px 20px rgba(18, 61, 36, 0.3);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.6s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(18, 61, 36, 0.4);
    background: linear-gradient(135deg, #235e35, #2d7a46);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Feature inputs */
.feature-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;
    margin-bottom: 6px;
}

.feature-number {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #17472a;
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 900;
    box-shadow: 0 4px 12px rgba(23, 71, 42, 0.3);
    transition: all 0.3s ease;
}

.feature-number:hover {
    transform: scale(1.1) rotate(5deg);
    background: #2d7a46;
}

.feature-name {
    color: #173d25;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.3px;
}

.feature-help {
    color: #718175;
    font-size: 11px;
}

/* Number input styling */
.stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(45, 93, 49, 0.2) !important;
    border-radius: 12px !important;
    color: #173d25 !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stNumberInput > div > div > input:hover {
    background: rgba(255, 255, 255, 1) !important;
    border-color: rgba(174, 224, 47, 0.5) !important;
}

.stNumberInput > div > div > input:focus {
    background: rgba(255, 255, 255, 1) !important;
    border-color: #aee02f !important;
    box-shadow: 0 0 20px rgba(174, 224, 47, 0.3) !important;
}

.stNumberInput > label {
    display: none;
}

/* Result card */
.result-card {
    margin-top: 25px;
    padding: 30px;
    text-align: center;
    background: linear-gradient(135deg, #e8f7d4, #ffffff);
    border: 2px solid #b9df58;
    border-radius: 24px;
    animation: result-reveal 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(174, 224, 47, 0.1) 0%, transparent 60%);
    animation: rotate-gradient 15s linear infinite;
}

@keyframes rotate-gradient {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes result-reveal {
    0% {
        opacity: 0;
        transform: scale(0.8) translateY(20px);
    }
    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

.result-card .model-label {
    font-size: 11px;
    font-weight: 800;
    color: #718175;
    letter-spacing: 2px;
    position: relative;
    z-index: 1;
}

.result-card .result-text {
    font-size: 44px;
    font-weight: 900;
    color: #1c602d;
    margin: 12px 0;
    position: relative;
    z-index: 1;
    animation: result-bounce 0.6s ease-out;
}

@keyframes result-bounce {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.result-card .confidence-text {
    font-size: 18px;
    font-weight: 700;
    color: #315f36;
    position: relative;
    z-index: 1;
}

/* Confidence bar */
.confidence-bar {
    width: 100%;
    height: 10px;
    background: rgba(45, 93, 49, 0.1);
    border-radius: 10px;
    margin-top: 15px;
    overflow: hidden;
    position: relative;
    z-index: 1;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #7cb342, #aee02f);
    border-radius: 10px;
    animation: confidence-grow 1.5s ease-out;
}

@keyframes confidence-grow {
    0% { width: 0%; }
}

/* Model info card */
.info-card {
    margin-top: 0;
    height: 100%;
}

.model-item {
    padding: 16px;
    background: #f2f8eb;
    border-radius: 16px;
    margin-top: 12px;
    border: 1px solid rgba(45, 93, 49, 0.1);
    transition: all 0.3s ease;
}

.model-item:hover {
    transform: translateX(5px);
    background: #e8f5df;
}

.model-label {
    font-size: 10px;
    font-weight: 900;
    color: #728074;
    letter-spacing: 1.5px;
}

.model-value {
    font-size: 15px;
    font-weight: 800;
    color: #173d25;
    margin-top: 4px;
}

.tip-box {
    margin-top: 20px;
    padding: 16px;
    background: #f3fae6;
    border-left: 4px solid #9fd433;
    border-radius: 10px;
    color: #3e5d43;
    animation: tip-pulse 4s ease-in-out infinite;
}

@keyframes tip-pulse {
    0%, 100% { box-shadow: 0 0 15px rgba(159, 212, 51, 0.15); }
    50% { box-shadow: 0 0 25px rgba(159, 212, 51, 0.25); }
}

.card-title {
    font-size: 22px;
    font-weight: 900;
    color: #173d25;
}

.card-subtitle {
    font-size: 13px;
    color: #718175;
    margin-top: 6px;
}

/* Footer */
.footer-text {
    text-align: center;
    margin-top: 45px;
    padding: 20px;
    color: #55745a;
    font-size: 13px;
    animation: fade-up 1s ease-out;
}

.footer-text b {
    color: #315f36;
}

/* Form card */
.form-card {
    background: rgba(255, 255, 255, 0.87);
    border-radius: 28px;
    padding: 24px;
    box-shadow: 0 14px 35px rgba(37, 82, 42, 0.12);
    margin-bottom: 12px;
    animation: fade-up 0.8s ease-out 0.2s both;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.3);
}

::-webkit-scrollbar-thumb {
    background: rgba(174, 224, 47, 0.5);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(174, 224, 47, 0.7);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .hero-title {
        font-size: 42px;
    }
    
    .stats-row {
        gap: 15px;
    }
    
    .stat-box {
        width: 130px;
        padding: 18px 12px;
    }
    
    .stat-value {
        font-size: 24px;
    }
}

/* Success animation */
@keyframes success-pulse {
    0% { box-shadow: 0 0 0 0 rgba(185, 223, 88, 0.7); }
    70% { box-shadow: 0 0 0 20px rgba(185, 223, 88, 0); }
    100% { box-shadow: 0 0 0 0 rgba(185, 223, 88, 0); }
}

.result-card.success {
    animation: success-pulse 1s ease-out, result-reveal 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

/* Main content layout */
.main-content {
    display: flex;
    gap: 25px;
    margin-top: 20px;
}

.main-content .measurements-col {
    flex: 1.3;
}

.main-content .info-col {
    flex: 0.7;
}
</style>
""", unsafe_allow_html=True)




# ---------- PARTICLES ----------
st.markdown("""
<div class="particles">
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>
""", unsafe_allow_html=True)




# ---------- FIRST PAGE / HERO ----------
current_time = datetime.now().strftime("%H:%M")


hero_html = f"""
<div class="hero">
    <div class="logo-wrap">
        <div class="pistachio-svg">{PISTACHIO_LOGO_SVG}</div>
    </div>
    <div class="badge">⭐ AI-Powered Classification</div>
    <div class="hero-title">Pistachio<span>Classification System</span></div>
    <div class="hero-description">
        Advanced machine learning classification of Kirmizi and Siirt pistachio
        varieties using 16 morphological measurements with state-of-the-art
        Gradient Boosting technology.
    </div>
    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-value">GB</div>
            <div class="stat-label">Algorithm</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">16</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">2</div>
            <div class="stat-label">Varieties</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{current_time}</div>
            <div class="stat-label">Session</div>
        </div>
    </div>
</div>
"""


st.markdown(hero_html, unsafe_allow_html=True)


# ---------- TITLE ----------
st.markdown('<div class="section-heading">Enter Pistachio Measurements</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheading">Use real values from image analysis for accurate classification.</div>', unsafe_allow_html=True)


# ---------- RESET BUTTON ----------
st.button("↻ Reset Values", on_click=reset_all_values, key=f"reset_btn_{st.session_state.animation_trigger}")


# ---------- MAIN CONTENT: Measurements + Model Info Side by Side ----------
main_col1, main_col2 = st.columns(2, gap="large")


with main_col1:
    # ---------- MEASUREMENTS SECTION ----------
    st.markdown("""
    <div class="form-card">
        <div class="card-title">📊 Enter Measurements</div>
        <div class="card-subtitle">16 morphological features from pistachio analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------- INPUTS ----------
    left_col, right_col = st.columns(2, gap="medium")
    
    for i, feature in enumerate(FEATURES):
        current_col = left_col if i % 2 == 0 else right_col
    
        with current_col:
            st.markdown(
                f'<div class="feature-label">'
                f'<span class="feature-number">{i + 1}</span>'
                f'<div><div class="feature-name">{feature}</div>'
                f'<div class="feature-help">{FEATURE_HELP[feature]}</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )
    
            st.number_input(
                feature,
                min_value=0.0,
                step=0.0001,
                format="%.4f",
                key=feature,
                label_visibility="collapsed"
            )
    
    st.write("")
    
    # ---------- CLASSIFY ----------
    if st.button("🔮 Classify Pistachio", key=f"classify_btn_{st.session_state.animation_trigger}"):
        # Check if all values are filled
        empty_fields = [f for f in FEATURES if st.session_state[f] == 0.0]
        
        if empty_fields:
            st.error(f"⚠️ Please enter values for: {', '.join(empty_fields)}")
        else:
            st.session_state.is_loading = True
            
            with st.spinner(text=''):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Simulate loading animation
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("Loading model...")
                        elif i < 60:
                            status_text.text("Processing features...")
                        elif i < 90:
                            status_text.text("Running classification...")
                        else:
                            status_text.text("Finalizing...")
                    
                    model, scaler, encoder = load_model_files()
        
                    input_values = [st.session_state[feature] for feature in FEATURES]
                    input_df = pd.DataFrame([input_values], columns=FEATURES)
        
                    scaled_input = scaler.transform(input_df)
                    prediction = model.predict(scaled_input)
                    probabilities = model.predict_proba(scaled_input)
        
                    predicted_label = encoder.inverse_transform(prediction)[0]
                    confidence = float(max(probabilities[0]) * 100)
        
                    st.session_state.result = predicted_label
                    st.session_state.confidence = confidence
        
                except Exception as error:
                    st.error(f"Prediction error: {error}")
                finally:
                    st.session_state.is_loading = False
                    progress_bar.empty()
                    status_text.empty()
    
    if st.session_state.result is not None:
        st.markdown(
            f'<div class="result-card success">'
            f'<div class="model-label">PREDICTED PISTACHIO VARIETY</div>'
            f'<div class="result-text">{st.session_state.result}</div>'
            f'<div class="confidence-text">Confidence: {st.session_state.confidence:.2f}%</div>'
            f'<div class="confidence-bar">'
            f'<div class="confidence-fill" style="width: {st.session_state.confidence}%;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


with main_col2:
    # ---------- MODEL INFO + GUIDANCE ----------
    st.html("""
    <div style="background: rgba(255, 255, 255, 0.87); border: 1px solid rgba(124, 179, 66, 0.3); border-radius: 28px; padding: 26px; box-shadow: 0 14px 35px rgba(37, 82, 42, 0.14); margin-bottom: 20px;">
        <div style="font-size: 22px; font-weight: 900; color: #173d25; margin-bottom: 8px;">ℹ️ Model Information</div>
        <div style="font-size: 13px; color: #718175; margin-bottom: 16px;">Classification details</div>
        
        <div style="padding: 16px; background: #f2f8eb; border-radius: 16px; margin-bottom: 12px; border: 1px solid rgba(45, 93, 49, 0.1);">
            <div style="font-size: 10px; font-weight: 900; color: #728074; letter-spacing: 1.5px;">🧠 ALGORITHM</div>
            <div style="font-size: 15px; font-weight: 800; color: #173d25; margin-top: 4px;">Gradient Boosting Classifier</div>
        </div>
        
        <div style="padding: 16px; background: #f2f8eb; border-radius: 16px; margin-bottom: 12px; border: 1px solid rgba(45, 93, 49, 0.1);">
            <div style="font-size: 10px; font-weight: 900; color: #728074; letter-spacing: 1.5px;">⚙️ PREPROCESSING</div>
            <div style="font-size: 15px; font-weight: 800; color: #173d25; margin-top: 4px;">StandardScaler</div>
        </div>
        
        <div style="padding: 16px; background: #f2f8eb; border-radius: 16px; margin-bottom: 12px; border: 1px solid rgba(45, 93, 49, 0.1);">
            <div style="font-size: 10px; font-weight: 900; color: #728074; letter-spacing: 1.5px;">🎯 OUTPUT</div>
            <div style="font-size: 15px; font-weight: 800; color: #173d25; margin-top: 4px;">Kirmizi or Siirt</div>
        </div>
        
        <div style="padding: 16px; background: #f2f8eb; border-radius: 16px; margin-bottom: 16px; border: 1px solid rgba(45, 93, 49, 0.1);">
            <div style="font-size: 10px; font-weight: 900; color: #728074; letter-spacing: 1.5px;">📊 FEATURES</div>
            <div style="font-size: 15px; font-weight: 800; color: #173d25; margin-top: 4px;">16 Morphological</div>
        </div>
        
        <div style="padding: 16px; background: #f3fae6; border-left: 4px solid #9fd433; border-radius: 10px; color: #3e5d43;">
            💡 <b>Tip:</b> Enter real measurements from pistachio image analysis for accurate classification.
        </div>
    </div>
    
    <div style="background: #f3fae6; border-radius: 16px; padding: 20px; border: 1px solid #c4e777;">
        <div style="font-size: 14px; font-weight: 800; color: #315f36; margin-bottom: 12px;">📋 Example Values (for guidance):</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; color: #3e5d43;">
            <div><b>AREA:</b> 85831.0000</div>
            <div><b>PERIMETER:</b> 1185.4930</div>
            <div><b>MAJOR_AXIS:</b> 448.8785</div>
            <div><b>MINOR_AXIS:</b> 244.3473</div>
            <div><b>ECCENTRICITY:</b> 0.8389</div>
            <div><b>EQDIASQ:</b> 330.5804</div>
            <div><b>SOLIDITY:</b> 0.9823</div>
            <div><b>CONVEX_AREA:</b> 87377.0000</div>
            <div><b>EXTENT:</b> 0.7485</div>
            <div><b>ASPECT_RATIO:</b> 1.8371</div>
            <div><b>ROUNDNESS:</b> 0.7675</div>
            <div><b>COMPACTNESS:</b> 0.7365</div>
            <div><b>SHAPEFACTOR_1:</b> 0.0052</div>
            <div><b>SHAPEFACTOR_2:</b> 0.0028</div>
            <div><b>SHAPEFACTOR_3:</b> 0.5424</div>
            <div><b>SHAPEFACTOR_4:</b> 0.9964</div>
        </div>
    </div>
    """)


st.markdown("""
<div class="footer-text">
    🥜 <b>Pistachio Classification System</b><br>
    Built with Python, Streamlit and Machine Learning
</div>
""", unsafe_allow_html=True)