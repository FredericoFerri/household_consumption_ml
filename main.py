from src.data.load_data import load_raw_data
from src.data.cleaning import clean_dataset
from src.data.preprocess import preprocess_features

from src.models.train_linear_regression import train_linear_regression
from src.models.evaluate import evaluate_model

def main():

    df = load_raw_data(
        "data/raw/household_energy_data.csv"
    )
    print("Dataset loaded.")

    df = clean_dataset(df)
    print("Dataset cleaned.")

    X, y = preprocess_features(df)
    print("Features preprocessed.")

    model, scaler, y_pred, y_test = train_linear_regression(X, y)
    print("Model trained.")

    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()