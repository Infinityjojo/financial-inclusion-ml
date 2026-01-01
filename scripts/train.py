# scripts/train.py
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocess import build_preprocessor

def train_model(df, target_col, model_dir="../models"):
    # DROP ID COLUMN
    df = df.drop(columns=["uniqueid"])

    X = df.drop(columns=[target_col])
    y = df[target_col]


    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    preprocessor = build_preprocessor(X)

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(pipeline, f"{model_dir}/bank_account_pipeline.pkl")
    joblib.dump(label_encoder, f"{model_dir}/target_encoder.pkl")

    return pipeline, X_test, y_test
