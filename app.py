import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
import joblib
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText

# ------------------- CONFIG -----------------------
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")
model = joblib.load("loan_default_model.pkl")

# ------------------- SIDEBAR INPUT ----------------
st.sidebar.header("📄 Upload Loan Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    try:
        pdf_text = ""
        for page in PyPDF2.PdfReader(uploaded_file).pages:
            pdf_text += page.extract_text()
        st.sidebar.success("✅ PDF uploaded successfully.")
    except:
        st.sidebar.error("❌ Error reading the PDF file.")

st.sidebar.header("🧮 Applicant Input Features")
age = st.sidebar.slider("Age", 18, 75, 30)
income = st.sidebar.number_input("Income", value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", value=10000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
months_employed = st.sidebar.slider("Months Employed", 0, 480, 24)
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 1, 10, 3)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, 10.0)
loan_term = st.sidebar.slider("Loan Term (months)", 12, 120, 36)
dti_ratio = st.sidebar.slider("DTI Ratio", 0.0, 1.0, 0.3)
education = st.sidebar.selectbox("Education", ["High School", "Master's", "PhD"])
employment_type = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
has_mortgage = st.sidebar.selectbox("Has Mortgage?", ["No", "Yes"])
has_dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Business", "Education", "Home", "Other"])
has_cosigner = st.sidebar.selectbox("Has Co-Signer?", ["No", "Yes"])

# ------------------- DATA FRAME -------------------
data = {
    "Age": age, "Income": income, "LoanAmount": loan_amount, "CreditScore": credit_score,
    "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate, "LoanTerm": loan_term, "DTIRatio": dti_ratio,
    "Education_High School": 1 if education == "High School" else 0,
    "Education_Master's": 1 if education == "Master's" else 0,
    "Education_PhD": 1 if education == "PhD" else 0,
    "EmploymentType_Part-time": 1 if employment_type == "Part-time" else 0,
    "EmploymentType_Self-employed": 1 if employment_type == "Self-employed" else 0,
    "EmploymentType_Unemployed": 1 if employment_type == "Unemployed" else 0,
    "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
    "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
    "HasMortgage_Yes": 1 if has_mortgage == "Yes" else 0,
    "HasDependents_Yes": 1 if has_dependents == "Yes" else 0,
    "LoanPurpose_Business": 1 if loan_purpose == "Business" else 0,
    "LoanPurpose_Education": 1 if loan_purpose == "Education" else 0,
    "LoanPurpose_Home": 1 if loan_purpose == "Home" else 0,
    "LoanPurpose_Other": 1 if loan_purpose == "Other" else 0,
    "HasCoSigner_Yes": 1 if has_cosigner == "Yes" else 0
}
df = pd.DataFrame([data])

# ------------------- HEADER -----------------------
st.title("📊 Loan Default Prediction App")
st.markdown("Welcome to the **Loan Default Prediction App**. Upload your loan document and enter the applicant’s details to check the risk of loan default using a trained machine learning model.")

st.markdown("---")
st.markdown("📁 **Project:** Data Science Lab  \n👨‍🎓 **Student:** Kummara Harish Kumar  \n👨‍🏫 **Professor:** Jan Lorenz")

# ------------------- PREDICTION -------------------
prediction = model.predict(df)[0]
confidence = model.predict_proba(df)[0][1]

if prediction == 1:
    prediction_text = f"⚠️ Risk: Likely to DEFAULT (Confidence: {confidence:.2f})"
    st.error(prediction_text)
else:
    prediction_text = f"✅ Safe: Not Likely to Default (Confidence: {1 - confidence:.2f})"
    st.success(prediction_text)

# ------------------- REPAYMENT --------------------
monthly_rate = interest_rate / 100 / 12
months = loan_term
monthly_payment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
total_payment = monthly_payment * months
repayment_text = f"**Monthly Repayment:** €{monthly_payment:.2f}  \n**Total Repayment:** €{total_payment:.2f} over {months} months"

st.markdown("### 💳 Loan Repayment Overview")
st.info(repayment_text)

# ------------------- PLOT -------------------------
st.markdown("### 📈 Credit Score Comparison")
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.barh(["Applicant", "Benchmark"], [credit_score, 700], color=["skyblue", "lightgreen"])
ax.set_xlim(300, 850)
st.pyplot(fig)

# ------------------- INPUT TABLE ------------------
st.markdown("### 📝 Applicant Input Summary")
st.dataframe(df.T.rename(columns={0: "Value"}))

# ------------------- PDF REPORT -------------------
def create_pdf(file_name, prediction_text, repayment_text, input_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Remove emojis before writing to PDF
    safe_prediction = prediction_text.replace("✅", "").replace("⚠️", "")
    safe_repayment = repayment_text.replace("€", "EUR")

    pdf.multi_cell(0, 10, f"Prediction: {safe_prediction}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Repayment Summary:\n{safe_repayment}")
    pdf.ln(5)

    pdf.cell(0, 10, "Applicant Details:", ln=True)
    for k, v in input_data.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)

    pdf.output(file_name)

st.markdown("### 📤 Export or Email Report")
email = st.text_input("Enter applicant's email address:", value="harishvadayar77@gmail.com")

if st.button("📄 Generate & Download PDF"):
    create_pdf("loan_report.pdf", prediction_text, repayment_text, df.loc[0].to_dict())
    with open("loan_report.pdf", "rb") as f:
        st.download_button("⬇️ Download PDF", f, file_name="loan_report.pdf")

if st.button("📧 Send Report to Email"):
    if email:
        create_pdf("loan_report.pdf", prediction_text, repayment_text, df.loc[0].to_dict())
        msg = MIMEMultipart()
        msg["From"] = "your_email@example.com"
        msg["To"] = email
        msg["Subject"] = "Loan Default Prediction Report"
        msg.attach(MIMEText("Attached is your loan prediction report.", "plain"))

        with open("loan_report.pdf", "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename="loan_report.pdf")
            msg.attach(part)

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login("your_email@example.com", "your_password")  # Replace
                server.send_message(msg)
            st.success("✅ Email sent successfully!")
        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")
    else:
        st.warning("⚠️ Please enter a valid email address.")

# ------------------- FOOTER -----------------------
st.markdown("---")
st.markdown("✅ Built with ❤️ by **Kummara Harish Kumar**  \n🧪 Project: **Data Science Lab**  \n👨‍🏫 Professor: **Jan Lorenz**")





















