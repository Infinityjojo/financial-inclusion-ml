# scripts/main.py
import pandas as pd
from train import train_model
from evaluate import evaluate_model


def main():
    df = pd.read_csv("../Data/Financial_inclusion_dataset.csv")

    model, X_test, y_test = train_model(
        df,
        target_col="bank_account"
    )

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
