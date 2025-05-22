# 📊 Loan Default Prediction App

This project is designed to predict whether a borrower is likely to default on a loan, using machine learning techniques applied to financial and demographic data. It includes a web application interface and detailed analysis in a Jupyter notebook.

## 🚀 Project Features

- 📈 **Exploratory Data Analysis**: Performed in `Analysis.ipynb` with insights into key features influencing loan default.
- 🧠 **Machine Learning Model**: A trained classification model (not uploaded due to GitHub size restrictions) predicts loan default.
- 🌐 **Flask Web Application**: `app.py` allows users to input borrower details and get a default risk prediction.
- 🧪 **Feature Importance Visualization**: Identifies key drivers of default such as income, interest rate, and loan amount.
- 📦 **Pickle Files**: Used to store expected columns (`expected_columns.pkl`) and support model inference.

## 📁 Repository Structure

📦 loan-default-prediction-app
┣ 📂 unzipped
┃ ┗ 📄 Loan_default.csv
┣ 📄 .gitignore
┣ 📄 app.py
┣ 📄 Analysis.ipynb
┣ 📄 expected_columns.pkl
┣ 📄 loan_report.pdf


## 🛠 How to Run

1. Clone the repository:
```bash
git clone https://github.com/vadaayar/loan-default-prediction-app.git
cd loan-default-prediction-app

pip install -r requirements.txt
python app.py

The model file loan_default_model.pkl is excluded due to GitHub's file size limits.

You can train your own model using the Analysis.ipynb.
