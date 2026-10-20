# 🎬 Movie Review Sentiment Analyzer

An NLP project that classifies movie reviews as **positive** or **negative** in real time.
Built for my Data Science internship final project.

**Live app:** _paste Streamlit link here_
**Demo video:** _paste Drive link here_

## Problem statement
Reading thousands of reviews by hand is slow. This project trains a machine-learning model that
predicts review sentiment automatically and serves it through a simple web app.

## Dataset
[IMDB Dataset of 50K Movie Reviews](https://zenodo.org/record/7929635) (Maas et al., 2011) -
50,000 reviews, 25,000 positive and 25,000 negative.

## Approach
1. Clean text (remove HTML tags, punctuation, lowercase)
2. TF-IDF features (unigrams + bigrams, 50k features)
3. Compare Logistic Regression, Naive Bayes and Linear SVM
4. Deploy the Logistic Regression model with Streamlit (it gives probability scores)

## Results
See `assets/results.csv` after running `train.py`, plus the charts in `assets/`.

## Project structure
```
├── app.py               # Streamlit web app
├── train.py             # trains, evaluates and saves the model
├── notebooks/           # step-by-step notebook
├── models/              # saved model (created by train.py)
├── assets/              # charts and results table
├── data/                # put IMDB Dataset.csv here
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
# put "IMDB Dataset.csv" inside data/
python train.py
streamlit run app.py
```

## Future work
DistilBERT / BERT fine-tuning, neutral class, sarcasm handling, Roman Urdu and multilingual
support, and tweet data.

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit
