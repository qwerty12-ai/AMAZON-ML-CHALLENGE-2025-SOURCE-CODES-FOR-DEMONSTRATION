# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** PrimeVision  
**Team Lead:** Mohd abdul sabeeh
## **Team Members:**
- Mohammad Sufiyan
- Venkat Pavan Kotwal Bakth
- Praveen Kumar

**Submission Date:** 13 Oct 25, 06:28 PM IST

---

## 1. Executive Summary
We developed a robust regression pipeline that predicts e-commerce product prices using only textual product data (catalog_content). Our solution relies on TF-IDF vectorization combined with handcrafted text-based features and a tuned LightGBM regressor. With minimal complexity, it achieved a SMAPE score of 50.5 on the public leaderboard.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

We developed a robust regression pipeline that predicts e-commerce product prices using only textual product data (catalog_content). Our solution relies on TF-IDF vectorization combined with handcrafted text-based features and a tuned LightGBM regressor. With minimal complexity, it achieved a SMAPE score of 50.5 on the public leaderboard.

**Key Observations:**

- Many catalog entries were noisy and unstructured.

- Price values had long-tail distribution → log-transform helped.

- Some high-frequency n-grams (e.g., "pack of", "ml", "combo") correlated with price.

### 2.2 Solution Strategy

**Approach Type:** Single Model(Text-only)

**Core Innovation:** 
A TF-IDF + handcrafted features pipeline with LightGBM regression. Despite not using images, we extracted linguistic signals like word count, punctuation, uppercase usage, and digit ratios which approximated product complexity or quantity.

---

## 3. Model Architecture

### 3.1 Architecture Overview

catalog_content ─┬─> Clean Text ──> TF-IDF(1-3 grams)
                 └─> Feature Extractor (len, digits, punct, etc.)
TF-IDF + Features ──> HStack ──> LightGBM Regressor ──> log(price) ──> expm1(pred)

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing: Lowercasing, punctuation removal, whitespace normalization.
- Vectorizer: TfidfVectorizer(ngram_range=(1,3), stop_words='english', max_features=20000)
- Extra features:

    - Text length
    - Word count
    - Digit count
    - Punctuation account
    - Uppercase character count
    - Average word length

- Model: LightGBM Regressor

    - n_estimators: 2000
    - learning_rate: 0.01
    - num_leaves: 127
    - subsample: 0.8
    - colsample_bytree: 0.8
    - Loss Function: RSME(trained on log(price))

---


## 4. Model Performance

### 4.1 Validation Results
- **Validation Split:** 10% of training data
- **SMAPE Score:** 50.56154261
- **Other Metrics:**
    - MAE(log): ~0.28
    - RSME(log): ~0.39


## 5. Conclusion
We built a simple yet effective pricing model using only the catalog text. Despite skipping image data, we reached close to leaderboard top scores through careful feature engineering and model tuning. We learned that meaningful text preprocessing and structured handcrafted features can rival more complex multimodal setups when time and resources are limited.
---


## Appendix

### A. Code artefacts

Submission ZIP (includes):

#### train_model.py

```python
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


#### sample_code.py

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

if __name__ == "_main_":
    main()


---
Other files:
model.joblib, vectorizer.joblib

Output: test_out.csv
---
