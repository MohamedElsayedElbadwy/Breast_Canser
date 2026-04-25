# Breast Cancer Diagnostic System (Deep Learning)

This project is a clinical decision support tool designed to assist in the classification of breast tumors as either **Malignant** or **Benign**. By leveraging an Artificial Neural Network (ANN), the system analyzes biopsy features with high precision to provide immediate diagnostic insights.

## Project Overview
Early detection is the most critical factor in successful cancer treatment. This application provides a streamlined, end-to-end pipeline that takes raw clinical parameters, processes them through a professional-grade preprocessing layer, and uses a trained Deep Learning model to generate a prediction.

## Key Features
* **Deep Learning Engine:** Built using a Keras/TensorFlow Neural Network.
* **Interactive Interface:** A fully responsive web dashboard powered by Streamlit.
* **Real-time Preprocessing:** Integrated data scaling to ensure prediction accuracy remains consistent with the training environment.
* **Visualization:** Detailed performance metrics and confidence scoring for every diagnosis.

## Technical Architecture
The system follows a modular architecture:
1.  **Data Layer:** Wisconsin Breast Cancer (Diagnostic) dataset.
2.  **Preprocessing:** Feature scaling using `StandardScaler` to normalize clinical inputs.
3.  **Inference:** A multi-layer perceptron (ANN) that processes 19 key tumor characteristics.
4.  **UI Layer:** Streamlit-based frontend for user-friendly interaction.

## How to Run
To run this project locally, ensure you have Python installed, then follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install streamlit tensorflow pandas scikit-learn joblib numpy matplotlip seaborn
