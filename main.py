from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from fastapi.templating import Jinja2Templates
import numpy as np


templates = Jinja2Templates(directory="templates")
model = joblib.load("model.pkl")
scaler  = joblib.load("scaler.pkl")

class Loan_system(BaseModel):
    dependents : int
    education : int
    self_emp : int
    income_annum : float
    loan_amount : float
    loan_term : float
    cibil_score : float
    residential : float
    commercial : float
    luxury : float
    bank : float

app = FastAPI()

@app.get("/")
def home(request : Request):
    return templates.TemplateResponse(name = "index.html", request=request)


@app.post("/predict")
def predict(data:Loan_system):
    data = np.array([[
        data.dependents,
        data.education,
        data.self_emp,
        data.income_annum,
        data.loan_amount,
        data.loan_term,
        data.cibil_score,
        data.residential,
        data.commercial,
        data.luxury,
        data.bank
    ]])

    data = scaler.transform(data)
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        return {"prediction": "Loan Approved"}
    else:
        return {"prediction": "Loan Rejected"}