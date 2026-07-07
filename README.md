# 🥜 Pistachio Classification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent web application that classifies pistachio varieties using machine learning. The system can accurately distinguish between **Kirmizi** and **Siirt** pistachio types based on morphological features.

## 🌟 Features

- **AI-Powered Classification**: Uses Gradient Boosting Classifier for accurate variety detection
- **Interactive Web Interface**: Beautiful Streamlit UI with real-time predictions
- **Visual Feedback**: Animated logo, smooth transitions, and intuitive design
- **16 Morphological Features**: Analyzes area, perimeter, shape factors, and more
- **Confidence Scoring**: Shows prediction confidence for each classification
- **History Tracking**: Keeps track of all past predictions with timestamps
- **Responsive Design**: Works seamlessly across desktop and mobile devices

## 🧠 How It Works

The system classifies pistachios based on 16 morphological features:

| Feature | Description |
|---------|-------------|
| AREA | Total area in pixels |
| PERIMETER | Outer boundary length |
| MAJOR_AXIS | Longest diameter of the nut |
| MINOR_AXIS | Shortest diameter of the nut |
| ECCENTRICITY | How elongated the shape is |
| EQDIASQ | Equivalent circle diameter |
| SOLIDITY | Area to convex area ratio |
| CONVEX_AREA | Convex hull area |
| EXTENT | Area to bounding box ratio |
| ASPECT_RATIO | Major to minor axis ratio |
| ROUNDNESS | Circularity measure |
| COMPACTNESS | Compactness indicator |
| SHAPEFACTOR_1-4 | Normalized shape ratios |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ChigurlaAparna/PISTACHIO_ML_PROJECT.git
   cd PISTACHIO_ML_PROJECT
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501` to use the application.

## 📊 Model Details

### Algorithm
- **Gradient Boosting Classifier**: Ensemble learning method that builds models sequentially
- Each tree corrects errors from previous trees, leading to high accuracy

### Preprocessing
- **Feature Scaling**: StandardScaler for normalized input features
- **Label Encoding**: LabelEncoder for target class mapping

### Performance
The model is pre-trained and saved as `pistachio_gradient_boosting_model.pkl` for instant predictions.

## 🎨 Screenshots

### Home Page
- Beautiful animated pistachio logo
- Quick stats showing model capabilities
- Easy-to-use feature input sliders

### Prediction Results
- Clear classification result (Kirmizi or Siirt)
- Confidence percentage display
- Feature contribution analysis

## 📁 Project Structure

```
PISTACHIO_ML_PROJECT/
├── app.py                           # Main Streamlit application
├── pistachio_gradient_boosting_model.pkl  # Trained ML model
├── pistachio_label_encoder.pkl      # Label encoder
├── pistachio_scaler.pkl            # Feature scaler
├── requirements.txt                 # Dependencies
├── runtime.txt                     # Python version
├── .python-version                 # Python version specification
└── README.md                       # This file
```

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web application framework |
| Scikit-learn | Machine learning library |
| Pandas | Data manipulation |
| Joblib | Model serialization |

## 💡 Usage Tips

1. **Enter Feature Values**: Use the sliders to input the 16 morphological features
2. **Click Classify**: Press the button to get the prediction
3. **View Results**: See the classification result with confidence score
4. **Track History**: View all past predictions in the History section

## 📝 Example Values

For testing, here are sample values from the dataset:

| Feature | Sample Value |
|---------|-------------|
| AREA | 85831 |
| PERIMETER | 1185.49 |
| MAJOR_AXIS | 448.88 |
| MINOR_AXIS | 244.35 |
| ECCENTRICITY | 0.84 |
| EQDIASQ | 330.58 |
| SOLIDITY | 0.98 |
| CONVEX_AREA | 87377 |
| EXTENT | 0.75 |
| ASPECT_RATIO | 1.84 |
| ROUNDNESS | 0.77 |
| COMPACTNESS | 0.74 |

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Built with ❤️ by [Chigurla Aparna](https://github.com/ChigurlaAparna)

## 🙏 Acknowledgments

- Dataset source and pistachio classification research
- Streamlit for the amazing web framework
- Scikit-learn for machine learning tools

---

<div align="center">

Made with 🥜 and Machine Learning

</div>
