"""
train_voice_model.py
Trains two real, saved scikit-learn models on the synthetic command
dataset:
  1. intent_model: classifies a sentence as GOTO / FETCH / STOP
  2. target_model: classifies WHICH destination or object is meant

Both are TF-IDF (character + word n-grams) + Logistic Regression,
which is fast to train (seconds, no GPU needed) and generalizes well
to unseen phrasings of a known place/object name, unlike hardcoded
keyword matching.

Run:
    python train_voice_model.py

Produces: voice_model.joblib
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from generate_training_data import build_dataset


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def train():
    data = build_dataset()
    texts = [row[0] for row in data]
    intents = [row[1] for row in data]
    # For target classification, only FETCH/GOTO rows have a meaningful
    # target; STOP rows are excluded from that model.
    target_rows = [(t, tgt) for (t, intent, tgt) in data if tgt is not None]
    target_texts = [r[0] for r in target_rows]
    target_labels = [r[1] for r in target_rows]

    X_tr, X_te, y_tr, y_te = train_test_split(texts, intents, test_size=0.2, random_state=0, stratify=intents)
    intent_model = make_pipeline()
    intent_model.fit(X_tr, y_tr)
    intent_acc = accuracy_score(y_te, intent_model.predict(X_te))

    Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
        target_texts, target_labels, test_size=0.2, random_state=0, stratify=target_labels
    )
    target_model = make_pipeline()
    target_model.fit(Xt_tr, yt_tr)
    target_acc = accuracy_score(yt_te, target_model.predict(Xt_te))

    # Refit on ALL data for the saved model (held-out sets above were
    # only to report honest accuracy numbers).
    intent_model_final = make_pipeline()
    intent_model_final.fit(texts, intents)

    target_model_final = make_pipeline()
    target_model_final.fit(target_texts, target_labels)

    joblib.dump(
        {"intent_model": intent_model_final, "target_model": target_model_final},
        "voice_model.joblib",
    )

    print(f"Intent model held-out accuracy: {intent_acc*100:.1f}%")
    print(f"Target model held-out accuracy: {target_acc*100:.1f}%")
    print(f"Trained on {len(texts)} intent examples, {len(target_texts)} target examples.")
    print("Saved -> voice_model.joblib")


if __name__ == "__main__":
    train()
