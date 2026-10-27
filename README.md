# ⚙️ Predictive Maintenance Using Machine Learning

## 📌 Project Overview

Predictive Maintenance is a machine learning project that predicts potential machine failures using the **AI4I 2020 Predictive Maintenance Dataset**. It compares two classification algorithms, Random Forest and K-Nearest Neighbors (KNN), to analyze their predictive performance.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and an interactive Streamlit web application.

## 🎯 Objectives

- Predict whether a machine may experience failure.
- Perform exploratory data analysis and visualize dataset patterns.
- Apply feature engineering and preprocessing techniques.
- Train Random Forest and KNN classification models.
- Compare models using standard evaluation metrics.
- Build an interactive web interface using Streamlit.

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains machine operating information, including:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type
- Machine failure (target variable)

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 🤖 Machine Learning Models

### 1. Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to classify machine failures.

### 2. K-Nearest Neighbors (KNN)
A classification algorithm that predicts a class based on the nearest training examples. StandardScaler is used to scale the input features.

## 🔍 Project Workflow

1. Load the dataset.
2. Perform exploratory data analysis.
3. Handle duplicate records and missing values.
4. Create additional features, including temperature difference and power proxy.
5. Encode categorical variables.
6. Split the data into training and testing sets.
7. Train Random Forest and KNN models.
8. Evaluate and compare model performance.
9. Save the trained models and supporting preprocessing files.
10. Deploy an interactive Streamlit frontend.

## 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

## 🌐 Streamlit Web Application

The application allows users to:

- Enter machine operating parameters.
- Generate predictions using both models.
- View estimated failure probabilities.
- Compare Random Forest and KNN predictions.
- View model performance metrics and charts.

## 📁 Repository Structure

```text
PredictiveMaintenance/
├── app.py
├── rf_model.pkl
├── knn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── fill_values.pkl
├── model_metrics.pkl
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the Streamlit application

```bash
streamlit run app.py
```

Ensure all required model and preprocessing files are present in the project directory.

## 📦 Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/) by connecting the GitHub repository and selecting `app.py` as the main file.

## ⚠️ Disclaimer

Predictions are based on the training dataset and are intended for educational and demonstration purposes. They are not a guarantee of actual machine failure.

## 👨‍💻 Author

**Pranjal Mishra**

Machine Learning Project — Predictive Maintenance

---

⭐ If you find this project useful, consider giving the repository a star.
