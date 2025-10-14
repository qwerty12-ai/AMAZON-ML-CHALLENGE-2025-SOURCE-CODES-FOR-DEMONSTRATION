import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
import joblib
from scipy.sparse import hstack

def extract_text_features(df):
    text = df['catalog_content'].fillna('')
    return pd.DataFrame({
        'text_len': text.str.len(),
        'word_count': text.str.split().apply(len),
        'num_digits': text.str.count(r'\d'),
        'num_punct': text.str.count(r'[^\w\s]'),
        'num_upper': text.str.count(r'[A-Z]'),
        'avg_word_len': text.apply(lambda x: np.mean([len(w) for w in x.split()]) if x else 0),
    })

def main():
    FOLDER = '../dataset/'
    train = pd.read_csv(os.path.join(FOLDER, 'train.csv'))

    # Remove price outliers (top 0.5%)
    price_cap = train['price'].quantile(0.995)
    train = train[train['price'] < price_cap]

    train['price_log'] = np.log1p(train['price'])

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 3), stop_words='english'
    )
    X_text = vectorizer.fit_transform(train['catalog_content'].fillna(''))

    extra_feats = extract_text_features(train)
    X_extra = extra_feats.values

    X = hstack([X_text, X_extra])
    y = train['price_log']

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()