import pandas as pd
df = pd.read_csv('dataset.csv')


def clean_text(text):
    text = text.lower().strip()
    return text
df['Description'] = df['Description'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['Description'])

from sklearn.linear_model import LogisticRegression

y = df['Category']
model = LogisticRegression()
model.fit(x,y)
    
def prediction(text):
    text_clean = clean_text(text)
    return vectorizer.transform([text_clean])

run = model.predict(prediction('ajio'))
print(run)

import joblib

# Save model and vectorizer
joblib.dump((model, vectorizer), 'predictor_ai.joblib')
