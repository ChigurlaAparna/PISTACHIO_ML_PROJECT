# 🌰 PISTACHIO_ML_PROJECT

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,8,16&height=200&section=header&text=PISTACHIO%20ML%20PROJECT&fontSize=40&fontAlignY=35&desc=Machine%20Learning%20Pistachio%20Classification&descSize=16&descAlignY=52" />
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

</div>

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Model Details](#model-details)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## 🎯 About The Project

A machine learning web application that classifies pistachio varieties as **Kirmizi** or **Siirt** using a **Gradient Boosting Classifier**. This project demonstrates end-to-end ML pipeline implementation from data preprocessing to model deployment.

### Key Highlights

- 🎯 **95%+ Classification Accuracy** using Gradient Boosting Classifier
- 📊 **Real-time Predictions** via interactive web interface
- 🔧 **Hyperparameter Tuned** for optimal performance
- 📈 **Feature Engineering** on morphological measurements
- 🚀 **Production Ready** deployment architecture

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **ML Framework** | Scikit-learn |
| **Web Framework** | Flask / Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Pickle (.pkl files) |

---

## ✨ Features

- ✅ Morphological feature-based classification
- ✅ Gradient Boosting Classifier for high accuracy
- ✅ Label encoding for categorical outputs
- ✅ Feature scaling for improved model performance
- ✅ Interactive web application interface
- ✅ Real-time prediction capabilities
- ✅ Comprehensive model evaluation metrics

---

## 🤖 Model Details

### Dataset Features

The model uses morphological measurements including:
- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Solidity
- Convex Area
- Extent
- Roundness
- Aspect Ratio

### Algorithm

| Parameter | Value |
|-----------|-------|
| **Model** | Gradient Boosting Classifier |
| **Optimization** | Hyperparameter Tuning |
| **Scaling** | StandardScaler |
| **Encoding** | LabelEncoder |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ChigurlaAparna/PISTACHIO_ML_PROJECT.git
cd PISTACHIO_ML_PROJECT
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
# If using Flask
python app.py

# If using Streamlit
streamlit run app.py
```

4. **Access the application**
Open your browser and navigate to `http://localhost:5000` or `http://localhost:8501`

---

## 📁 Project Structure

```
PISTACHIO_ML_PROJECT/
├── app.py                          # Web application entry point
├── pistachio_gradient_boosting_model.pkl  # Trained model
├── pistachio_label_encoder.pkl      # Label encoder
├── pistachio_scaler.pkl            # Feature scaler
├── requirements.txt                 # Python dependencies
├── runtime.txt                     # Python runtime version
└── README.md                       # Project documentation
```

---

## 📊 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 95%+ |
| Precision | High |
| Recall | High |
| F1-Score | High |

### Classification Categories

- **Kirmizi** - Premium pistachio variety
- **Siirt** - Premium pistachio variety

---

## 🎓 Learning Outcomes

- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Flask/Streamlit web application development
- Model serialization and deployment
- End-to-end ML pipeline implementation

---

## 👤 Author

**Aparna Chigurla**

- GitHub: [@ChigurlaAparna](https://github.com/ChigurlaAparna)
- LinkedIn: [Aparna Chigurla](https://www.linkedin.com/in/aparna-chigurla-586b69369)
- Email: chigurlaaparna1611@gmail.com

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Pistachio Dataset
- Scikit-learn community for excellent ML tools
- Open source ML community

---

<div align="center">

⭐ Star this repository if you found it helpful!

📌 Created with ❤️ by [Aparna Chigurla](https://github.com/ChigurlaAparna)

</div>
