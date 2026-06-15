import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# Load Model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #1e3c72;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    background-color: #1e88e5;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #1565c0;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">🏦 AI Loan Approval System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enter customer details to check loan eligibility</div>',
    unsafe_allow_html=True
)

# Input Layout
col1, col2 = st.columns(2)

with col1:
    dependents = st.number_input(
        "👨‍👩‍👧 Number of Dependents",
        min_value=0,
        step=1
    )

    education = st.selectbox(
        "🎓 Education",
        ["Graduate", "Not Graduate"]
    )

    income = st.number_input(
        "💰 Annual Income",
        min_value=0.0
    )

    loan_amount = st.number_input(
        "🏦 Loan Amount",
        min_value=0.0
    )

    loan_term = st.number_input(
        "📅 Loan Term",
        min_value=0.0
    )

with col2:
    self_employed = st.selectbox(
        "💼 Self Employed",
        ["Yes", "No"]
    )

    cibil = st.number_input(
        "⭐ CIBIL Score",
        min_value=0.0
    )

    residential = st.number_input(
        "🏠 Residential Assets",
        min_value=0.0
    )

    commercial = st.number_input(
        "🏢 Commercial Assets",
        min_value=0.0
    )

    luxury = st.number_input(
        "🚗 Luxury Assets",
        min_value=0.0
    )

    bank = st.number_input(
        "💳 Bank Assets",
        min_value=0.0
    )

# Encoding
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Prediction
if st.button("🔍 Predict Loan Status"):

    data = np.array([[
        dependents,
        education,
        self_employed,
        income,
        loan_amount,
        loan_term,
        cibil,
        residential,
        commercial,
        luxury,
        bank
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.markdown("---")

    if prediction[0] == 1:
        st.balloons()
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")