from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load model and vectorizer once at startup
model, vectorizer = joblib.load('predictor_ai.joblib')
amount =0
title = ""
type1 = ["Expense"]
def clean_text(text):
    return text.lower().strip()

# Request model
class TextRequest(BaseModel):
    text: str

# Create app
app = FastAPI()

@app.post("/predict")
def predict_category(data: TextRequest):
    if not data.text:
        raise HTTPException(status_code=400, detail="Text field is required.")
    
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    for i in cleaned.split(' '):
        if i.isnumeric():
            amount = i
    
    income_words = [
    "income", "earning", "salary", "wage", "paycheck", "stipend", "bonus", "revenue",
    "profit", "gain", "inflow", "deposit", "earning", "commission", "return", "dividend",
    "interest", "yield", "payout", "proceeds", "net income", "gross income",
    "remittance", "royalty", "passive income", "freelance income", "consulting fee",
    "rental income", "side hustle", "earnings", "compensation", "benefit",
    "monetization", "credit", "cash inflow", "business income", "capital gain",
    "gratuity", "provident fund", "incentive", "retainer", "honorarium", "reimbursement",
    "salary credit", "income transfer", "monthly income", "quarterly earnings",
    "annual income", "income source", "app earnings", "pension", "alimony",
    "scholarship", "financial aid", "tax refund"
    ]


    for i in cleaned.split(' '):
        if i in income_words:
            type1[0]="Income"
            break
        else:
            type1[0]="Expense"

    return {"Title":vectorizer,"prediction": prediction,"Amount":amount,"Type":type1[0]}
