from fastapi import FastAPI, HTTPException
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

@app.post("/predict")
def predict_category(data: TextRequest):
    if not data.text:
        raise HTTPException(status_code=400, detail="Text field is required.")
    
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    return {"prediction": prediction}
