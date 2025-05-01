from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load model and vectorizer once at startup
model, vectorizer = joblib.load('predictor_ai.joblib')


def clean_text(text):
    return text.lower().strip()

# Request model
class TextRequest(BaseModel):
    text: str

# Create app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:4001",
    "https://your-production-frontend.com",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_category(data: TextRequest):
    if not data.text:
        raise HTTPException(status_code=400, detail="Text field is required.")
    
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    amount =0
    title = "other"
    type1 = "expense"
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
    title_list = ['shop', 'swiggy', 'zomato', 'starbucks', 'pizza hut', 'mcdonalds', 
                  'dominos', 'burger king', 'subway', 'kfc', 'uber', 'ola', 'metro', 
                  'auto', 'train', 'cab', 'bus', 'amazon', 'flipkart', 'myntra', 
                  'big bazaar', 'ajio', 'nykaa', 'meesho', 'snapdeal', 'electricity', 
                  'water bill', 'gas bill', 'mobile recharge', 'broadband bill', 'wifi', 
                  'dth recharge', 'insurance', 'rent']

    for i in cleaned.split(' '):
        if i.isnumeric() ==True:
            amount = i
            break
    
    

    for i in cleaned.split(' '):
        if i in income_words:
            type1="income"
            break
        else:
            type1="expense"

    for i in cleaned.split(' '):
        if i in title_list:
            title=i
            break

    return {"title":title,"prediction": prediction,"amount":amount,"type":type1}
