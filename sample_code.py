import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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
    test_df = pd.read_csv(os.path.join(FOLDER, 'test.csv'))
    test_output_file = 'test_out.csv'

    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')

    X_text = vectorizer.transform(test_df['catalog_content'].fillna(''))
    extra_feats = extract_text_features(test_df)
    X_extra = extra_feats.values

    X = hstack([X_text, X_extra])

    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0.5, None)

    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': preds
    })

    output_df.to_csv(os.path.join(FOLDER, test_output_file), index=False)
    print('Predictions saved to', test_output_file)
    print(output_df.head())

if __name__ == "__main__":
    main()